"""Implements the core evolution algorithm."""
from __future__ import print_function


from neat.six_util import iteritems, itervalues
from neat.math_util import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from explaneat.core.backprop import NeatNet

## Replace neat-based reporting with explaneat extensions of the reporting
## methods with hooks regarding backprop
# from neat.reporting import ReporterSet
# from neat.reporting import BaseReporter
from explaneat.core.experiment import ExperimentReporterSet as ReporterSet


# from explaneat.core.experiment import

from neat.population import Population


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
                optimizer = optim.Adadelta,
                nEpochs = 100):
        self.reporters = ReporterSet()
        self.config = config

        if not type(xs) is torch.Tensor:
            self.xs = torch.tensor(xs)
        else:
            self.xs = xs
        if not type(ys) is torch.Tensor:
            self.ys = torch.tensor(ys)
        else:
            self.ys = ys

        self.optimizer = optimizer
        self.criterion = criterion

        self.nEpochs = nEpochs

        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
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
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def backpropagate(self, xs, ys, nEpochs = 300):
        try:
            nEpochs = self.config.generations_of_backprop
        except AttributeError:
            nEpochs = nEpochs
        losses = []
        postLosses = []
        improvements = []
        for k, genome in self.population.items():

            net = NeatNet(genome, self.config, criterion=self.criterion)
            
            preBPLoss = net.meanLoss(xs, ys)
            
            net.optimise(xs, ys, nEpochs)
            
            postBPLoss = net.meanLoss(xs, ys)
            postLosses.append(postBPLoss)
            
            lossDiff = postBPLoss - preBPLoss

            losses.append((preBPLoss, postBPLoss, lossDiff))
            improvements.append(lossDiff)
            net.updateGenomeWeights(genome)

        print('mean improvement: %s' % mean(improvements))
        print('best improvement: %s' % min(improvements))
        print('best loss: %s' % min(postLosses))
            

    def run(self, fitness_function, n=None, nEpochs = 100):
        """
        
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            self.reporters.pre_backprop(self.config, self.population, self.species)
            
            self.backpropagate(self.xs, self.ys, nEpochs=nEpochs)

            self.reporters.post_backprop(self.config, self.population, self.species)
            
            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.reporters.pre_reproduction(self.config, self.population, self.species)
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            self.reporters.post_reproduction(self.config, self.population, self.species)


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
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1
        
        self.reporters.end_experiment(self.config, self.population, self.species)

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome


