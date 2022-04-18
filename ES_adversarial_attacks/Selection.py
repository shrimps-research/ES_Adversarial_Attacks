import numpy as np
from ES_adversarial_attacks.Population import Population


class Selection:
    def __call__(self):
        pass


class PlusSelection(Selection):
    """ Get the best individuals from both the parent and offspring populations
    """
    def __call__(self, parents: Population, offspring: Population, minimize=True):
        if minimize:
            sorted_ind = np.argsort(np.hstack([parents.fitnesses, offspring.fitnesses]))[:parents.pop_size]
        else:
            sorted_ind = np.argsort(np.hstack([parents.fitnesses, offspring.fitnesses]))[::-1][:parents.pop_size]
        parents.individuals = np.vstack([parents.individuals, offspring.individuals])[sorted_ind]
        parents.fitnesses = list(np.hstack([parents.fitnesses, offspring.fitnesses])[sorted_ind])
        if not parents.mutation.__class__.__name__ == "OneSigma":
            parents.sigmas = np.vstack([parents.sigmas, offspring.sigmas])[sorted_ind]
        else:
            parents.sigmas = np.hstack([parents.sigmas, offspring.sigmas])[sorted_ind]


class CommaSelection(Selection):
    """ Get the best individuals from the offspring population
    """
    def __call__(self, parents: Population, offspring: Population, minimize=True):
        if minimize:
            sorted_ind = np.argsort(offspring.fitnesses)[:parents.pop_size]
        else:
            sorted_ind = np.argsort(offspring.fitnesses)[::-1][:parents.pop_size]
        parents.individuals = offspring.individuals[sorted_ind]
        parents.sigmas = offspring.sigmas[sorted_ind]
        parents.fitnesses = list(np.array(offspring.fitnesses)[sorted_ind])