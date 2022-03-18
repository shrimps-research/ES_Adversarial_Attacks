from classes.Individual import Individual
from classes.Population import Population
import numpy as np

class Recombination:
    def recombine(self, population: Population):
        pass

class Intermediate(Recombination):
    """
    Creates offspring by taking the average values of the parents
    """
    def __init__(self,offspring_size):
        self.offspring_size = offspring_size
    
    
    def recombine(self, parent: Population) -> Population:
        idxes = np.arange(0,parent.size)
        offspring_population = Population(self.offspring_size, parent.n_values)
        i = 0
        while i < self.offspring_size:
            # Pick two parents at random
            parents_idx = np.random.choice(idxes,2,replace=False)

            # Create offspring 
            offspring_values = np.average([ parent.individuals[parents_idx[0]].values,
                                            parent.individuals[parents_idx[1]].values],
                                            axis=0)
            offspring_sigmas = np.average([ parent.individuals[parents_idx[0]].sigmas,
                                            parent.individuals[parents_idx[1]].sigmas],
                                            axis=0)
            offspring_alphas = np.average([ parent.individuals[parents_idx[0]].alphas,
                                            parent.individuals[parents_idx[1]].alphas],
                                            axis=0)

            offspring = Individual(parent.n_values)
            offspring.values = offspring_values
            offspring.sigmas = offspring_sigmas
            offspring.alphas = offspring_alphas
            offspring_population.individuals[i] = offspring

            i += 1
        
        return offspring_population

        
