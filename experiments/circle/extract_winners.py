from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np

from visualize import draw_net

import os
import sys

import neat

import random

import visualize

def getFinalCheckpoint(dirPath):
    checkpoints = os.listdir(dirPath)
    for checkpoint in checkpoints:
        if 'final' in checkpoint:
            return os.path.join(dirPath, checkpoint)
    return None


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


# def eval_genomes(genome, config):
#     net = neat.nn.FeedForwardNetwork.create(genome, config)

#     error = 400.0
#     net = neat.nn.FeedForwardNetwork.create(genome, config)
#     for xi, xo in zip(inputs, outputs):
#         output = net.activate(xi)
#         error -= (output[0] - xo[0]) ** 2
#     error = error/100
#     return error


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 400.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2
        genome.fitness = genome.fitness/100.0
        
if __name__ == "__main__":
    for runNo in range(10):

        checkpoint = getFinalCheckpoint('./results/run-%s' % runNo)
        # checkpoint = './results/run-0/circle-checkpoint-993'
        print(checkpoint)

        p = neat.Checkpointer.restore_checkpoint(checkpoint)
        p.generation = int(p.generation[:-6])
        # print(p.checkpoint)

        # pe1 = neat.ParallelEvaluator(4, eval_genomes)
        winner = p.run(eval_genomes, 1)

        node_names = {-1:'A', -2: 'B', 0:'OUT'}

        d = visualize.draw_net(p.config, winner, False, node_names=node_names, filename='winner_%s.svg'%runNo)

