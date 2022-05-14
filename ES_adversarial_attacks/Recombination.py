from ES_adversarial_attacks.Population import Population
import numpy as np
import random


class Recombination:
    def __call__(self):
        pass


class Intermediate(Recombination):
    """ Creates offspring by taking the average values of the parents
    """    
    def __call__(self, parents: Population, offspring: Population):
        offspring.individuals = []
        offspring.sigmas = []
        for _ in range(offspring.pop_size):
            # get pair
            i1, i2 = random.sample(range(parents.individuals.shape[0]), k=2)
            x1, x2 = parents.individuals[i1], parents.individuals[i2]
            s1, s2 = parents.sigmas[i1], parents.sigmas[i2]
            # recombinate components and sigmas
            offspring.individuals.append(np.array([(x1_i + x2_i) / 2 for x1_i, x2_i in zip(x1, x2)]))
            if not parents.mutation.__class__.__name__ == "OneSigma":
                # select the sigma of each component as the mean of the sigmas of the two components
                offspring.sigmas.append(np.array([(s1_i + s2_i) / 2 for s1_i, s2_i in zip(s1, s2)]))
            else:
                # sigma is the mean of the two sigmas
                offspring.sigmas.append((s1 + s2) / 2)
        offspring.individuals = np.vstack(offspring.individuals)
        if not parents.mutation.__class__.__name__ == "OneSigma":
            offspring.sigmas = np.vstack(offspring.sigmas)
        else:
            offspring.sigmas = np.array(offspring.sigmas)


class Discrete(Recombination):
    """ Creates discrete recombined offsprings.
    """
    def __call__(self, parents: Population, offspring: Population):
        offspring.individuals = []
        offspring.sigmas = []
        for _ in range(offspring.pop_size):
            # get pair
            i1, i2 = random.sample(range(parents.individuals.shape[0]), k=2)
            x1, x2 = parents.individuals[i1], parents.individuals[i2]
            s1, s2 = parents.sigmas[i1], parents.sigmas[i2]
            # probabilities of selecting each component (and/or sigma) of x1
            prob_x1 = np.random.uniform(0, 1, size=offspring.ind_dim)
            # recombinate components and sigmas
            offspring.individuals.append(np.array([x1_i if prob_x1_i >= 0.5 else x2_i for x1_i, x2_i, prob_x1_i in zip(x1, x2, prob_x1)]))
            if not parents.mutation.__class__.__name__ == "OneSigma":
                # select each sigma identically to the components selection
                offspring.sigmas.append(np.array([s1_i if prob_s1_i >= 0.5 else s2_i for s1_i, s2_i, prob_s1_i in zip(s1, s2, prob_x1)]))
            else:
                # select at random one of the two sigmas
                offspring.sigmas.append(s1 if np.random.uniform(0, 1) >= 0.5 else s2)
        offspring.individuals = np.vstack(offspring.individuals)
        if not parents.mutation.__class__.__name__ == "OneSigma":
            offspring.sigmas = np.vstack(offspring.sigmas)
        else:
            offspring.sigmas = np.array(offspring.sigmas)