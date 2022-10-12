import argparse
import os
import datetime
import random
import json

from explaneat.experimenter.experiment import GenericExperiment
from explaneat.data.wranglers import GENERIC_WRANGLER
from explaneat.experimenter.results import Result
from explaneat.evaluators.evaluators import binary_cross_entropy


from explaneat.core.neuralneat import NeuralNeat as nneat
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
model_config = experiment.config['model']['neural_network']

# ---------------- Load data ------------------------------

processed_data_location = experiment.data_folder

generic_wrangler = GENERIC_WRANGLER(processed_data_location)

X_train, y_train = generic_wrangler.train_sets_as_np
X_test, y_test = generic_wrangler.test_sets_as_np

X_test_tt = torch.tensor(X_test)
y_test_tt = torch.tensor(y_test)


# ------------------- Set up environment ------------------------------

# epoch_points = [10]
config_path = experiment.config['model']['propneat']['base_config_path']
base_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)


# ------------------- Define model ------------------------------


def instantiate_population(config, xs, ys):

    # if not os.path.exists(saveLocation):
    # os.makedirs(saveLocation)

    # config.save(os.path.join(saveLocation, 'config.conf'))

    # Create the population, which is the top-level object for a NEAT run.
    p = BackpropPopulation(config,
                           xs,
                           ys,
                           criterion=nn.BCELoss())

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(
    # 5, filename_prefix=str(saveLocation) + "checkpoint-"))
    # bpReporter = backprop.BackpropReporter(True)
    # p.add_reporter(bpReporter)
    # p.add_reporter(ExperimentReporter(saveLocation))

    return p

# ------------------- instantiate model ------------------------------


# ------------------- train model ------------------------------
my_random_seed = experiment.config["random_seed"]

for iteration_no in range(1):
    my_random_seed = experiment.config["random_seed"] + iteration_no
    random.seed(my_random_seed)

    config = deepcopy(base_config)

    # saveLocation = saveLocationTemplate.format(
    # experiment.config['model']['random_forest'], iteration_no)

    p = instantiate_population(config, X_train, y_train)
    # Run for up to nGenerations generations.
    winner = p.run(binary_cross_entropy,
                   experiment.config['model']['propneat']["max_n_generations"],
                   nEpochs=experiment.config['model']['propneat']['epochs_per_generation'])

    g = p.best_genome

    explainer = ExplaNEAT(g, config)

    g_result = Result(
        g,
        "best_genome",
        experiment.config['experiment']['name'],
        experiment.config['data']['raw_location'],
        experiment.experiment_sha,
        iteration_no,
        {
            "iteration": iteration_no
        }
    )

    experiment.results_database.add_result(g_result)
    g_map = Result(
        visualize.draw_net(config, g).source,
        "best_genome_map",
        experiment.config['experiment']['name'],
        experiment.config['data']['raw_location'],
        experiment.experiment_sha,
        iteration_no,
        {
            "iteration": iteration_no,
        }
    )
    experiment.results_database.add_result(g_map)

    skippiness = Result(
        explainer.skippines(),
        "skippiness",
        experiment.config['experiment']['name'],
        experiment.config['data']['raw_location'],
        experiment.experiment_sha,
        iteration_no,
        {
            "iteration": iteration_no,
        }
    )
    experiment.results_database.add_result(skippiness)

    propneat_results_tt = explainer.net.forward(X_test_tt)
    propneat_results = [r[0] for r in propneat_results_tt.detach().numpy()]

    preds_results = Result(
        json.dumps(list(propneat_results)),
        "propneat_prediction",
        experiment.config['experiment']['name'],
        experiment.config['data']['raw_location'],
        experiment.experiment_sha,
        0,
        {
            "iteration": iteration_no
        }
    )
    experiment.results_database.add_result(preds_results)

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
