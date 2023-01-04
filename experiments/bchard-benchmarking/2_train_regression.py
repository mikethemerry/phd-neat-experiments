# Import the model we are using
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
X_test, y_test = generic_wrangler.test_sets


# ------------------- train model ------------------------------

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
regression_preds = [pred[0] for pred in regression_model.predict(X_test)]

preds_results = Result(
    json.dumps(list(regression_preds)),
    "linear_regression_predictions",
    experiment.config['experiment']['name'],
    experiment.config['data']['raw_location'],
    experiment.experiment_sha,
    0,
    {
        "iteration": 0
    }
)
experiment.results_database.add_result(preds_results)


regression_model = LogisticRegression()
regression_model.fit(X_train, y_train)
regression_preds = [pred[0] for pred in regression_model.predict(X_test)]

preds_results = Result(
    json.dumps(list(regression_preds)),
    "logistic_regression_predictions",
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
