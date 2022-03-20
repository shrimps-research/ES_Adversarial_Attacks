from classes.Population import Population
import numpy as np
import random


class Recombination:
    def __call__(self):
        pass


class Intermediate(Recombination):
    """ Creates offspring by taking the average values of the parents
    """
    def __init__(self, offspring_size):
        self.offspring_size = offspring_size
    
    def __call__(self, parents: Population, offspring: Population) -> Population:
        offspring.individuals = []
        offspring.sigmas = []
        for _ in range(self.offspring_size):
            # get pair
            i1, i2 = random.sample(range(parents.individuals.shape[0]), k=2)
            x1, x2 = parents.individuals[i1], parents.individuals[i2]
            s1, s2 = parents.sigmas[i1], parents.sigmas[i2]
            # recombinate components and sigmas
            offspring.individuals.append(np.array([(x1_i + x2_i) / 2 for x1_i, x2_i in zip(x1, x2)]))
            if not parents.one_sigma:
                # select the sigma of each component as the mean of the sigmas of the two components
                offspring.sigmas.append(np.array([(s1_i + s2_i) / 2 for s1_i, s2_i in zip(s1, s2)]))
            else:
                # sigma is the mean of the two sigmas
                offspring.sigmas.append((s1 + s2) / 2)
        offspring.individuals = np.vstack(offspring.individuals)
        if not parents.one_sigma:
            offspring.sigmas = np.vstack(offspring.sigmas)
        else:
            offspring.sigmas = np.array(offspring.sigmas)