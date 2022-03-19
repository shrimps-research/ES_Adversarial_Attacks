import numpy as np
from Population import Population

class Selection:
    def select(self, population: Population):
        pass


class OnePlusL(Selection):
    """
    Get the best individuals from both the parent and offspring populations
    """
    def select(self, parents: Population, offspring: Population):
        all_individuals = np.concatenate([parents.individuals, offspring.individuals])
        evals = np.concatenate([parents.all_fitnesses(), offspring.all_fitnesses()])

        indexes = evals.argsort()[:parents.size]
        parents.individuals = [all_individuals[idx] for idx in indexes]
        return parents


class OneCommaL(Selection):
    """
    Get the best individuals from the offspring population
    """
    def select(self, parents: Population, offspring: Population):
        evals = offspring.all_fitnesses()
        indexes = evals.argsort()[:parents.size]
        parents.individuals = [offspring.individuals[idx] for idx in indexes]
        return parents

