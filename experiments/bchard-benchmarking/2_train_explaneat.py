import argparse
import os
import datetime
import random

from explaneat.experimenter.experiment import GenericExperiment
from explaneat.data.wranglers import GENERIC_WRANGLER
from explaneat.experimenter.results import Result

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

X_train, y_train = generic_wrangler.train_sets
X_test, y_test = generic_wrangler.test_sets


train_data = generic_wrangler.Train_Dataset
train_loader = DataLoader(train_data,
                          batch_size=model_config["batch_size"],
                          shuffle=True)

validate_data = generic_wrangler.Test_Dataset
validate_loader = DataLoader(dataset=validate_data,
                             batch_size=model_config["batch_size"],
                             shuffle=True)

total_step = len(train_loader)

# ------------------- Set up environment ------------------------------

epoch_points = [10]

# ------------------- Define model ------------------------------


def instantiate_population(config, xs, ys, saveLocation):

    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)

    config.save(os.path.join(saveLocation, 'config.conf'))

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
    bpReporter = backprop.BackpropReporter(True)
    p.add_reporter(bpReporter)
    p.add_reporter(ExperimentReporter(saveLocation))

    return p

# ------------------- instantiate model ------------------------------


# ------------------- train model ------------------------------
my_random_seed = experiment.config["random_seed"]
for epochs in epoch_points:
    for iteration_no in range(1):
        my_random_seed += 1
        random.seed(my_random_seed)
        start_time = datetime.now()

        logger.info("################################################")
        logger.info("################################################")
        logger.info("Starting epochs {} iteration {}".format(
            epochs, iteration_no))
        logger.info("Started at {}".format(
            start_time.strftime("%m/%d/%Y, %H:%M:%S")))
        logger.info("################################################")
        logger.info("################################################")

        config = deepcopy(base_config)
#         config.pop_size = pop_size

        saveLocation = saveLocationTemplate.format(epochs, iteration_no)

        p = instantiate_population(
            config, data_wrangler.X_train, data_wrangler.y_train, saveLocation)
        # Run for up to nGenerations generations.
        winner = p.run(binary_cross_entropy,
                       experiment.config["max_n_generations"], nEpochs=epochs)

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
                "iteration": iteration_no,
                "n_epochs": epochs
            }
        )
        resultsDB.add_result(g_result)

        g_map = Result(
            visualize.draw_net(config, g).source,
            "best_genome_map",
            experiment.config['experiment']['name'],
            experiment.config['data']['raw_location'],
            experiment.experiment_sha,
            iteration_no,
            {
                "iteration": iteration_no,
                "n_epochs": epochs
            }
        )
        resultsDB.add_result(g_map)

        skippiness = Result(
            explainer.skippines(),
            "skippiness",
            experiment.config['experiment']['name'],
            experiment.config['data']['raw_location'],
            experiment.experiment_sha,
            iteration_no,
            {
                "iteration": iteration_no,
                "n_epochs": epochs
            }
        )
        resultsDB.add_result(skippiness)

        end_time = datetime.now()

        p.reporters.reporters[2].save_checkpoint(
            p.config, p.population, p.species, str(p.generation) + "-final")

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        results = []
        for xi, xo in zip(data_wrangler.X_test, data_wrangler.y_test):
            output = winner_net.activate(xi)
            results.append([xi, xo, output])

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(saveLocation, 'results.csv'))

        ancestry = p.reporters.reporters[3].trace_ancestry_of_species(
            g.key, p.reproduction.ancestors)

        ancestors = {
            k: v['genome'] for k, v in p.reporters.reporters[3].ancestry.items()
        }

        resultsDB.save()


# ------------------- get predictions ------------------------------
nn_preds = None

preds_results = Result(
    nn_preds,
    "nn_prediction",
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
