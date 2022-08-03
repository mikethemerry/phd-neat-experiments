import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

import pprint

from explaneat.core.neuralneat import NeuralNeat


class ExplaNEAT():
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.net = NeuralNeat(genome, config)
        self.phenotype = self.net

    def shapes(self):
        return self.net.shapes()

    def n_genome_params(self):
        nNodes = len(self.genome.nodes)
        nConnections = len(self.genome.connections)

        return nNodes + nConnections

    def density(self):
        nParams = self.n_genome_params()
        denseSize = 0
        for ix, shape in self.shapes().items():
            denseSize += shape[0]*shape[1]
        return nParams/denseSize

    def node_depth(self, nodeId):
        return self.net.node_mapping.node_mapping[nodeId]['depth']

    def skippines(self):
        skippy_sum = 0
        for connection in self.genome.connections:
            skippy = self.node_depth(
                connection[1]) - self.node_depth(connection[0]) - 1
            skippy_sum += skippy
        return skippy_sum/len(self.genome.connections)
