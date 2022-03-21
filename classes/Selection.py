import numpy as np
from classes.Population import Population


class Selection:
    def __call__(self):
        pass


# TODO add maximization case (argsort)
class PlusSelection(Selection):
    """ Get the best individuals from both the parent and offspring populations
    """
    def __call__(self, parents: Population, offspring: Population, minimize=True):
        sorted_ind = np.argsort(np.hstack([parents.fitnesses, offspring.fitnesses]))
        parents.individuals = np.vstack([parents.individuals, offspring.individuals])[sorted_ind][:parents.individuals.shape[0]]
        if not parents.one_sigma:
            parents.sigmas = np.vstack([parents.sigmas, offspring.sigmas])[sorted_ind][:parents.individuals.shape[0]]
        else:
            parents.sigmas = np.hstack([parents.sigmas, offspring.sigmas])[sorted_ind][:parents.individuals.shape[0]]


# TODO add maximization case (argsort)
class CommaSelection(Selection):
    """ Get the best individuals from the offspring population
    """
    def __call__(self, parents: Population, offspring: Population, minimize=True):
        sorted_ind = np.argsort(offspring.fitnesses)
        parents.individuals = offspring.individuals[sorted_ind][:parents.individuals.shape[0]]
        parents.sigmas = offspring.sigmas[sorted_ind][:parents.individuals.shape[0]]