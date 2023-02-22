# Import the model we are using
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import argparse
import os
import json

from explaneat.experimenter.experiment import GenericExperiment

from explaneat.data.wranglers import GENERIC_WRANGLER

from explaneat.experimenter.results import Result


parser = argparse.ArgumentParser(description="Provide the experiment config")
parser.add_argument('conf_file',
                    metavar='experiment_config_file',
                    type=str,
                    help="Path to experiment config")
parser.add_argument("ref_file",
                    metavar='experiment_reference_file',
                    type=str,
                    help="Path to experiment ref file")

args = parser.parse_args()

experiment = GenericExperiment(
    args.conf_file,
    confirm_path_creation=False,
    ref_file=args.ref_file)
logger = experiment.logger


experiment.create_logging_header("Starting {}".format(__file__), 50)

# ---------------- Load data ------------------------------

processed_data_location = experiment.data_folder

generic_wrangler = GENERIC_WRANGLER(processed_data_location)

X_train, y_train = generic_wrangler.train_sets
X_test = generic_wrangler.X_test

y_train = y_train.values.ravel()

# ------------------- Hyperparameter setup ------------------------------


model = RandomForestRegressor(random_state=experiment.random_seed)
param_vals = experiment.config['model']['random_forest']['hyperparameter_ranges']
random_rf = RandomizedSearchCV(estimator=model,
                               param_distributions=param_vals,
                               n_iter=experiment.config['hyperparam_tuning']['n_iter'],
                               scoring=experiment.config['hyperparam_tuning']['scoring'],
                               cv=experiment.config['hyperparam_tuning']['cv'],
                               refit=True,
                               n_jobs=-1,
                               random_state=experiment.config['random_seed'])

#Training and prediction

random_rf.fit(X_train, y_train)
preds = random_rf.best_estimator_.predict(X_test)

# Train the model on training data
random_rf.fit(X_train, y_train)
# Use the forest's predict method on the test data
rf_preds = random_rf.predict(X_test)


preds_results = Result(
    json.dumps(list(rf_preds)),
    "rf_predictions",
    experiment.config['experiment']['name'],
    experiment.config['data']['raw_location'],
    experiment.experiment_sha,
    0,
    {
        "iteration": 0
    }
)
experiment.results_database.add_result(preds_results)

experiment.create_logging_header("Ending {}".format(__file__), 50)
