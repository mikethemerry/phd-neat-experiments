"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import visualize

import pandas as pd
import numpy as np
import random


def xor(a, b):
    response = False
    if a > 0.5 and b < 0.5:
        response = True
    if a < 0.5 and b > 0.5:
        response = True
    # return (1.0, 0.0) if response else (0.0, 1.0)
    return 1.0 if response else 0.0
    

def create_n_points(n, size, min=0.0, max=1.0):
    data = []
    for _ in range(n):
        data.append(tuple([
            random.uniform(min, max) for ii in range(size)
        ]))

    return data

# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def overUnder(val, threshold):
    return 1. if val > threshold else 0

xor_inputs_2 = create_n_points(400, 2)

xor_outputs_2 = [
    tuple( [xor(tup[0], tup[1])] ) for tup in xor_inputs_2
]

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genomes(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    error = 4.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
    return error
def eval_genomes2(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    error = 400.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for xi, xo in zip(xor_inputs_2, xor_outputs_2):
        output = net.activate(xi)
        # output = softmax(output)
        output[0] = overUnder(output[0], 0.5)

        # genome.fitness -= ((output[0] - xo[0][0]) ** 2) + ((output[1] - xo[0][1]) ** 2)

        error -= ((output[0] - xo[0]) ** 2)
    error = error/100.0
    return error


def run(config_file, runNumber):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    try:
        os.mkdir('./results/run-%s' % runNumber)
    except FileExistsError:
        pass
    p.add_reporter(neat.Checkpointer(25, filename_prefix='./results/run-%s/xor-checkpoint-' % runNumber))

    # Run for up to 300 generations.
    pe1 = neat.ParallelEvaluator(4, eval_genomes)
    pe2 = neat.ParallelEvaluator(4, eval_genomes2)

    p.run(pe1.evaluate, 300)
    p.reporters.reporters[2].save_checkpoint(p.config, p.population, p.species, str(p.generation) + "-final-pretrain")  
    winner = p.run(pe2.evaluate, 1000)

    p.reporters.reporters[2].save_checkpoint(p.config, p.population, p.species, str(p.generation) + "-final")  
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    results = []
    for xi, xo in zip(xor_inputs_2, xor_outputs_2):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
        results.append([xi[0], xi[1], output[0]])

    df = pd.DataFrame(results)
    df.to_csv('results-run-%s.csv' % runNumber)

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    return p

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    for ii in range(3, 10):
        run(config_path, ii)