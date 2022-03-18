import numpy as np

from classes.Population import Population
from classes.Individual import Individual

class Mutation:
    def mutate(self, individual: Individual):
        pass

    def mutate_population(self, population: Population):
        """
        Mutates the whole population
        """
        for individual in population.individuals:
            self.mutate(individual)


class CustomSigma(Mutation):
    """
    Custom sigma method, different from the individual sigma done during the course.
    """

    def mutate(self, individual: Individual):
        """
        Mutates a single individual
        """
        lr = 1/np.sqrt(2*(np.sqrt(individual.n_values)))
        lr_prime = 1/(np.sqrt(2*individual.n_values))

        for curr_sig in range(individual.n_values):
            # Update current sigma
            normal_matr_prime = np.random.normal(0,lr_prime,1)
            normal_matr = np.random.normal(0,lr,1)
            individual.sigmas[curr_sig] = individual.sigmas[curr_sig]*(
                                    np.exp(normal_matr+normal_matr_prime))

            sigma_noise = np.random.normal(0,individual.sigmas[curr_sig],1)
            individual.values[curr_sig] = individual.values[curr_sig] + sigma_noise

    
