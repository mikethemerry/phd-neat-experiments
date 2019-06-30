"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat

import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim


from explaneat.core.backprop import NeatNet
from explaneat.core import backprop
from explaneat.core.backproppop import BackpropPopulation
from explaneat.visualization import visualize
from explaneat.core.experiment import ExperimentReporter

from sklearn import datasets
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

random.seed(4242)
nGenerations = 3
# random.seed(43)
def one_hot_encode(vals):
    width = max(vals)
    newVals = []
    for val in vals:
        blank = [0. for _ in range(width + 1)]
        blank[val] = 1.
        newVals.append(blank)
    return np.asarray(newVals)

iris = datasets.load_iris()
xs_raw = iris.data[:, :2]  # we only take the first two features.
scaler = StandardScaler()
scaler.fit(xs_raw)
xs = scaler.transform(xs_raw)
ys = iris.target
ys_onehot = one_hot_encode(ys)
# ys = one_hot_encode(iris.target)

def eval_genomes(genomes, config):
    loss = nn.CrossEntropyLoss()
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        preds = []
        for xi in xs:
            preds.append(net.activate(xi))
        # print("my predictions are")
        # print(torch.tensor(preds))
        # print(torch.tensor(ys))
        genome.fitness = float(1./loss(torch.tensor(preds), torch.tensor(ys)))
        # genome.fitness = metrics.log_loss(ys, preds)
        # genome.fitness -= (output[0] - xo[0]) ** 2



def run(config_file, runNumber):

    saveLocation = './../../data/experiments/iris/experiment-0/'
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)



    # Create the population, which is the top-level object for a NEAT run.
    p = BackpropPopulation(config, 
                            xs, 
                            ys, 
                            criterion=nn.CrossEntropyLoss())

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=saveLocation))
    bpReporter = backprop.BackpropReporter(True)
    p.add_reporter(bpReporter)
    p.add_reporter(ExperimentReporter(saveLocation))

    # Run for up to nGenerations generations.
    winner = p.run(eval_genomes, nGenerations)

    p.reporters.reporters[2].save_checkpoint(p.config, p.population, p.species, str(p.generation) + "-final")  
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    results = []
    for xi, xo in zip(xs, ys):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
        results.append([xi[0], xi[1], output])

    df = pd.DataFrame(results)
    df.to_csv('results.csv')

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)

    return p

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-feedforward')
    config_path = os.path.join(local_dir, 'config-iris')
    p = run(config_path, 0)

    g = p.best_genome

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = NeatNet(g, config) 

    winnerNet = neat.nn.FeedForwardNetwork.create(g, config)

    ancestry = p.reporters.reporters[3].trace_ancestry_of_species(g.key, p.reproduction.ancestors) 

    print('have ancestry')

    ancestors = {
        k: v['genome'] for k, v in p.reporters.reporters[3].ancestry.items()
    }
    print('have ancestors')
    visualize.create_ancestry_video(p.config, g, ancestry, ancestors, p.reporters.reporters[1])
    print('have finished video')