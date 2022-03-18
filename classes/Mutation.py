import numpy as np

from classes.Population import Population
from classes.Individual import Individual

class Mutation:
    def mutate(self, individual: Individual) -> np.ndarray:
        pass


class CustomSigma(Mutation):

    def __init__(self, lr, lr_prime):
        self.lr = lr
        self.lr_prime = lr_prime  

    def mutate(self, individual: Individual):
        for curr_sig in range(individual.n_values):
            # Update current sigma
            normal_matr_prime = np.random.normal(0,self.lr_prime,1)
            normal_matr = np.random.normal(0,self.lr,1)
            individual.sigmas[curr_sig] = individual.sigmas[curr_sig]*(
                                    np.exp(normal_matr+normal_matr_prime))

            sigma_noise = np.random.normal(0,individual.sigmas[curr_sig],1)
            individual.values[curr_sig] = individual.values[curr_sig] + sigma_noise
