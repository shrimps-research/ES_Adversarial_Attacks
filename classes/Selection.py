import numpy as np
from classes.Population import Population


class Selection:
    def __call__(self):
        pass


class PlusSelection(Selection):
    """ Get the best individuals from both the parent and offspring populations
    """
    def __call__(self, parents: Population, offspring: Population):
        sorted_ind = np.argsort(np.hstack([parents.fitnesses, offspring.fitnesses]))
        offspring.individuals = np.vstack([parents.individuals, offspring.individuals])[sorted_ind][:parents.individuals.shape[0]]
        if not parents.one_sigma:
            offspring.sigmas = np.vstack([parents.sigmas, offspring.sigmas])[sorted_ind][:parents.individuals.shape[0]]
        else:
            offspring.sigmas = np.hstack([parents.sigmas, offspring.sigmas])[sorted_ind][:parents.individuals.shape[0]]


class CommaSelection(Selection):
    """ Get the best individuals from the offspring population
    """
    def __call__(self, parents: Population, offspring: Population):
        sorted_ind = np.argsort(offspring.fitnesses)
        offspring.individuals = offspring.individuals[sorted_ind][:parents.individuals.shape[0]]
        offspring.sigmas = offspring.sigmas[sorted_ind][:parents.individuals.shape[0]]