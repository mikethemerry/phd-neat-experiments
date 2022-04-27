"""Implements the core evolution algorithm."""
from __future__ import print_function

import sys


from neat.math_util import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from explaneat.core.neuralneat import NeuralNeat as nneat


from explaneat.core.backprop import NeatNet

from explaneat.core.errors import GenomeNotValidError

# from explaneat.core.neuralneat import NeuralNeat

# Replace neat-based reporting with explaneat extensions of the reporting
# methods with hooks regarding backprop
# from neat.reporting import ReporterSet
# from neat.reporting import BaseReporter
from explaneat.core.experiment import ExperimentReporterSet as ReporterSet
from explaneat.core.utility import MethodTimer


# from explaneat.core.experiment import

from neat.population import Population

import logging


class BackpropPopulation(Population):
    """
    This class extends the core NEAT implementation with a backprop method
    """

    def __init__(self,
                 config,
                 xs,
                 ys,
                 initial_state=None,
                 criterion=nn.BCELoss(),
                 optimizer=optim.Adadelta,
                 nEpochs=100,
                 device=None):
        self.logger = logging.getLogger("experimenter.backproppop")
        self.reporters = ReporterSet()
        self.config = config

        USE_CUDA = torch.cuda.is_available()
        USE_CUDA = False
        self.device = torch.device("cuda:1" if USE_CUDA else "cpu")

        if not type(xs) is torch.Tensor:
            self.xs = torch.tensor(xs, dtype=torch.float64)
        else:
            self.xs = xs
        if not type(ys) is torch.Tensor:
            self.ys = torch.tensor(ys, dtype=torch.float64)
        else:
            self.ys = ys

        self.optimizer = optimizer
        self.criterion = criterion

        self.nEpochs = nEpochs

        stagnation = config.stagnation_type(
            config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)

        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(
                config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def backpropagate(self, xs, ys, nEpochs=5):
        self.logger.info(
            'about to start backprop with {} epochs'.format(nEpochs))
        try:
            nEpochs = self.config.generations_of_backprop
        except AttributeError:
            nEpochs = nEpochs
        losses = []
        postLosses = []
        improvements = []
        for k, genome in self.population.items():

            # net = NeatNet(genome, self.config, criterion=self.criterion)
            try:
                net = nneat(genome, self.config,
                            criterion=nn.BCEWithLogitsLoss())
            except GenomeNotValidError:
                ("This net - {} isn't valid".format(k))
                preBPLoss = 0
                postBPLoss = 99999
                lossDiff = postBPLoss - preBPLoss
                improvements.append(lossDiff)
                losses.append((preBPLoss, postBPLoss, lossDiff))
                postLosses.append(postBPLoss)
                continue

            optimizer = optim.Adadelta(net.parameters(), lr=1.5)

            optimizer.zero_grad()
            losses = []
            try:
                preBPLoss = F.mse_loss(net.forward(xs), ys).sqrt()
            except:
                net.help_me_debug()
                sys.exit("Error in loss for backprop")
            for i in range(nEpochs):
                preds = net.forward(xs)
                loss = F.mse_loss(preds, ys).sqrt()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss)
            # losses[-10:]
            postBPLoss = F.mse_loss(net.forward(xs), ys).sqrt()
            lossDiff = postBPLoss - preBPLoss

            losses.append((preBPLoss, postBPLoss, lossDiff))
            improvements.append(lossDiff.item())
            # self.logger.info(net.weights)
            # self.logger.info("PRE")
            # for ix in genome.connections:
            #     self.logger.info(genome.connections[ix])
            # self.logger.info(net.weights)
            net.update_genome_weights()  # Not updating?
            self.population[k] = net.genome
            # for ix in genome.connections:
            # self.logger.info(genome.connections[ix])
            # for ix in net.genome.connections:
            # self.logger.info(net.genome.connections[ix])
            postLosses.append(postBPLoss.item())

        self.logger.info('mean improvement: %s' % mean(improvements))
        self.logger.info('best improvement: %s' % min(improvements))
        self.logger.info('best loss: %s' % min(postLosses))

    def run(self, fitness_function, n=None, nEpochs=100):
        """

        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError(
                "Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            with MethodTimer('generationStart'):
                self.reporters.start_generation(self.generation)

            with MethodTimer('pre_backprop'):
                self.reporters.pre_backprop(
                    self.config, self.population, self.species)

            with MethodTimer('backprop'):
                self.backpropagate(self.xs, self.ys, nEpochs=nEpochs)

            with MethodTimer('post_backprop'):
                self.reporters.post_backprop(
                    self.config, self.population, self.species)

            logging.debug('The current population after backpropagation is')
            logging.debug(self.population)

            # Evaluate all genomes using the user-provided function.
            # fitness_function(list(iter(self.population.iteritems())), self.config)

            with MethodTimer('evaluate fitness'):
                fitness_function(self.population, self.config,
                                 self.xs, self.ys, self.device)

            # Gather and report statistics.
            best = None
            for genome_id, g in self.population.items():
                if best is None or g.fitness > best.fitness:
                    best = g
            with MethodTimer('post evaluate'):
                self.reporters.post_evaluate(
                    self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(
                    g.fitness for genome_id, g in self.population.items())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(
                        self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.

            with MethodTimer('pre_reproduction'):
                self.reporters.pre_reproduction(
                    self.config, self.population, self.species)

            with MethodTimer('reproduction'):
                self.population = self.reproduction.reproduce(self.config, self.species,
                                                              self.config.pop_size, self.generation)

            with MethodTimer('post reproduction'):
                self.reporters.post_reproduction(
                    self.config, self.population, self.species)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.

            with MethodTimer('speciate'):
                self.species.speciate(
                    self.config, self.population, self.generation)

            with MethodTimer('end generation'):
                self.reporters.end_generation(
                    self.config, self.population, self.species)

            self.generation += 1

        self.reporters.end_experiment(
            self.config, self.population, self.species)

        if self.config.no_fitness_termination:
            self.reporters.found_solution(
                self.config, self.generation, self.best_genome)

        return self.best_genome
