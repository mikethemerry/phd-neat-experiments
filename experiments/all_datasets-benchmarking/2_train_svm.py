from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

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
parser.add_argument('data_name', metavar='experiment_data_file', type=str,
                    help="Path to experiment data")

args = parser.parse_args()

experiment = GenericExperiment(
    args.conf_file,
    confirm_path_creation=False,
    ref_file=args.ref_file)
logger = experiment.logger


experiment.create_logging_header("Starting {}".format(__file__), 50)

# ---------------- Load data ------------------------------

base_data_location = experiment.data_folder
processed_data_location = os.path.join(base_data_location, args.data_name)

generic_wrangler = GENERIC_WRANGLER(processed_data_location)

X_train, y_train = generic_wrangler.train_sets
X_test, y_test = generic_wrangler.test_sets


# ------------------- train model ------------------------------

svm_model = SVC()
param_vals = experiment.config['model']['svm']['hyperparameter_ranges']


random_svm = RandomizedSearchCV(estimator=svm_model,
                                param_distributions=param_vals,
                                n_iter=experiment.config['hyperparam_tuning']['n_iter'],
                                scoring=experiment.config['hyperparam_tuning']['scoring'],
                                cv=experiment.config['hyperparam_tuning']['cv'],
                                refit=True,
                                n_jobs=-1,
                                random_state=experiment.config['random_seed'])


random_svm.fit(X_train, y_train)
svm_preds = random_svm.predict(X_test)

recast_preds = [float(p) for p in svm_preds]

preds_results = Result(
    json.dumps(list(recast_preds)),
    "svm_predictions",
    experiment.config['experiment']['name'],
    args.data_name,
    experiment.experiment_sha,
    0,
    {
        "iteration": 0
    }
)
experiment.results_database.add_result(preds_results)

experiment.create_logging_header("Ending {}".format(__file__), 50)
