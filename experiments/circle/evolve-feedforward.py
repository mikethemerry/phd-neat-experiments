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


def x3(a, b):
    return overUnder(b, a**3)

def inCircle(x, y, radius=0.8):
    return overUnder(x**2 + y**2, radius**2)


inputs = create_n_points(400, 2, -1, 1)

outputs = [
    tuple( [inCircle(tup[0], tup[1])] ) for tup in inputs
]

# # 2-input XOR inputs and expected outputs.
# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genomes(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    error = 400.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for xi, xo in zip(inputs, outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
    error = error/100
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
    p.add_reporter(neat.Checkpointer(25, filename_prefix='./results/run-%s/circle-checkpoint-' % runNumber))

    # Run for up to 300 generations.
    pe1 = neat.ParallelEvaluator(4, eval_genomes)

    winner = p.run(pe1.evaluate, 1000)
    p.reporters.reporters[2].save_checkpoint(p.config, p.population, p.species, str(p.generation) + "-final")  
    # winner = p.run(pe2.evaluate, 1000)

    # p.reporters.reporters[2].save_checkpoint(p.config, p.population, p.species, str(p.generation) + "-final")  
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(inputs, outputs):
        # output = winner_net.activate(xi)
        # print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    results = []
    for xi, xo in zip(inputs, outputs):
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
    for ii in range(0, 10):
        run(config_path, ii)