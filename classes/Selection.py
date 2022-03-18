import numpy as np
from classes.Population import Population

class Selection:
    def select(self, population: Population):
        pass


class OnePlusL(Selection):
    """
    Get the best individuals from both the parent and offspring populations
    """
    def select(self, parents: Population,offspring: Population):
        all_individuals = np.concatenate([parents,offspring])
        evals = np.concatenate([parents.all_fitnesses() ,offspring.all_fitnesses()])

        indexes = evals.argsort()[:parents.size]
        return [all_individuals[idx] for idx in indexes]


class OneCommaL(Selection):
    """
    Get the best individuals from the offspring population
    """
    def select(self, offspring: Population):
        evals = offspring.all_fitnesses()
        indexes = evals.argsort()[:offspring.size]
        return [offspring[idx] for idx in indexes]

