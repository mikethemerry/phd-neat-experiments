import argparse
import os
import datetime
import random
import json
import tempfile

from explaneat.experimenter.experiment import GenericExperiment
from explaneat.data.wranglers import GENERIC_WRANGLER
from explaneat.experimenter.results import Result
from explaneat.evaluators.evaluators import binary_cross_entropy

from sklearn.model_selection import train_test_split

# from explaneat.core.neuralneat import NeuralNeat as nneat
from explaneat.core.backprop import NeatNet as naiveNeat
from explaneat.core import backprop
from explaneat.core.backproppop import BackpropPopulation
from explaneat.visualization import visualize
from explaneat.core.experiment import ExperimentReporter
from explaneat.core.utility import one_hot_encode
from explaneat.core.explaneat import ExplaNEAT
from explaneat.experimenter.results import Result, ResultsDatabase

import neat


from copy import deepcopy


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description="Provide the experiment config")
parser.add_argument(
    "conf_file",
    metavar="experiment_config_file",
    type=str,
    help="Path to experiment config",
)
parser.add_argument(
    "ref_file",
    metavar="experiment_reference_file",
    type=str,
    help="Path to experiment ref file",
)
parser.add_argument(
    "data_name",
    metavar="experiment_data_file",
    type=str,
    help="Path to experiment data",
)

args = parser.parse_args()

experiment = GenericExperiment(
    args.conf_file, confirm_path_creation=False, ref_file=args.ref_file
)
logger = experiment.logger


experiment.create_logging_header("Starting {}".format(__file__), 50)
model_config = experiment.config["model"]["neural_network"]

# ---------------- Load data ------------------------------

base_data_location = experiment.data_folder
processed_data_location = os.path.join(base_data_location, args.data_name)


generic_wrangler = GENERIC_WRANGLER(processed_data_location)

X_train_base, y_train_base = generic_wrangler.train_sets_as_np
X_test, y_test = generic_wrangler.test_sets_as_np


X_test_tt = torch.tensor(X_test)
y_test_tt = torch.tensor(y_test)


# ------------------- Set up environment ------------------------------

config_path = experiment.config["model"]["propneat"]["base_config_path"]

# epoch_points = [10]
# Manually create temporary file in the same directory as the original file
temp_file_path = os.path.join(os.path.dirname(config_path), "temp_config.ini")
with open(temp_file_path, "w") as temp_file, open(config_path, "r") as original_file:
    # Copy contents of original file to temporary file
    temp_file.write(original_file.read())

    # Add two lines to the end of the temporary file
    temp_file.write("\nnum_inputs = {}".format(X_test_tt.shape[1]))

# Call the runFile function with the temporary file
base_config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    temp_file_path,
)

# Delete the temporary file
os.remove(temp_file_path)


base_config.pop_size = experiment.config["model"]["propneat"]["population_size"]


# base_config.genome_config.num_inputs = X_test_tt.shape[1]
# base_config.num_inputs = X_test_tt.shape[1]
# ------------------- Define model ------------------------------


def instantiate_population(config, xs, ys):
    # if not os.path.exists(saveLocation):
    # os.makedirs(saveLocation)

    # config.save(os.path.join(saveLocation, 'config.conf'))

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    return p


def eval_genomes(genomes, config):
    ## evaluate the genomes using binary cross entropy
    binary_cross_entropy(genomes, config, X_train, y_train)


# ------------------- instantiate model ------------------------------


# ------------------- train model ------------------------------
my_random_seed = experiment.config["random_seed"]

for iteration_no in range(experiment.config["model"]["propneat"]["n_iterations"]):
    my_random_seed = experiment.config["random_seed"] + iteration_no

    random.seed(my_random_seed)

    # split data into train and validate using sklearn
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train_base, y_train_base, test_size=0.3, random_state=my_random_seed
    )

    config = deepcopy(base_config)

    p = instantiate_population(config, X_train, y_train)
    # Run for up to nGenerations generations.
    winner = p.run(
        eval_genomes,
        experiment.config["model"]["propneat"]["max_n_generations"]
        * experiment.config["model"]["propneat"]["epochs_per_generation"],
    )

    g = p.best_genome

    explainer = ExplaNEAT(g, config, neat_class=naiveNeat)

    g_result = Result(
        g,
        "NEAT_best_genome",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {"iteration": iteration_no * 100},
    )

    experiment.results_database.add_result(g_result)
    g_map = Result(
        visualize.draw_net(config, g).source,
        "NEAT_best_genome_map",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {
            "iteration": iteration_no * 100,
        },
    )
    experiment.results_database.add_result(g_map)

    skippiness = Result(
        explainer.skippines(),
        "NEAT_skippiness",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {
            "iteration": iteration_no * 100,
        },
    )
    experiment.results_database.add_result(skippiness)

    depth = Result(
        explainer.depth(),
        "NEAT_depth",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {
            "iteration": iteration_no * 100,
        },
    )
    experiment.results_database.add_result(depth)

    param_size = Result(
        explainer.n_genome_params(),
        "NEAT_param_size",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {
            "iteration": iteration_no * 100,
        },
    )
    experiment.results_database.add_result(param_size)

    propneat_results_tt = explainer.net.forward(X_test_tt)
    propneat_results = [r[0] for r in propneat_results_tt.detach().numpy()]

    preds_results = Result(
        json.dumps(list(propneat_results)),
        "NEAT_prediction",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {"iteration": iteration_no * 100},
    )
    experiment.results_database.add_result(preds_results)

    experiment.results_database.save()

    experiment.create_logging_header("Ending {} - variation 1".format(__file__), 50)

    experiment.results_database.save()

    # end_time = datetime.now()

    # p.reporters.reporters[2].save_checkpoint(
    #     p.config, p.population, p.species, str(p.generation) + "-final")

    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # results = []
    # for xi, xo in zip(data_wrangler.X_test, data_wrangler.y_test):
    #     output = winner_net.activate(xi)
    #     results.append([xi, xo, output])

    # ancestry = p.reporters.reporters[3].trace_ancestry_of_species(
    #     g.key, p.reproduction.ancestors)

    # ancestors = {
    #     k: v['genome'] for k, v in p.reporters.reporters[3].ancestry.items()
    # }

    # resultsDB.save()


# ------------------- get predictions ------------------------------


experiment.create_logging_header("Ending {}".format(__file__), 50)


experiment.create_logging_header("Starting {} - variation 1".format(__file__), 50)


experiment.create_logging_header("Ending {} - variation 2".format(__file__), 50)
