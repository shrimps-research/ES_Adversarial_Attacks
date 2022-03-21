import numpy as np
import random
import math
from classes.Population import Population
from classes.Individual import Individual


class Mutation:
    def mutate(self):
        """ Mutates a single individual
        """
        pass

    def __call__(self, population: Population):
        """ Mutates the whole population
        """
        self.mutate(population)


class IndividualSigma(Mutation):
    """ Individual sigma method.
    """
    def mutate(self, population: Population):
        """ Mutates the population
        """
        tau = 1 / np.sqrt(2 * np.sqrt(population.individuals.shape[1]))
        tau_prime = 1 / np.sqrt(2 * population.individuals.shape[1])
        # one draw from N(0, tau') per individual
        tau_prime_drawns = np.random.normal(0, tau_prime, size=population.sigmas.shape[0]).reshape(-1, 1).repeat(population.sigmas.shape[1], axis=1)
        # one draw from N(0, tau) per sigma (individuals x components)
        tau_drawns = np.random.normal(0, tau, size=population.sigmas.shape)
        # mutate sigmas
        population.sigmas = population.sigmas * np.exp(tau_drawns + tau_prime_drawns)
        # mutate components
        variations = np.random.normal(0, population.sigmas)
        population.individuals += variations


# TODO fix the use of Individual class
class CustomSigma(Mutation):
    """ Custom sigma method, experiment with using random lr_prime for each sigma.
        Not very efficient, advised not to use.
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

            # Update individual's values
            sigma_noise = np.random.normal(0,individual.sigmas[curr_sig],1)
            individual.values[curr_sig] = individual.values[curr_sig] + sigma_noise


# TODO fix the use of Individual class
class Correlated(Mutation):
    def mutate(self, individual: Individual):
        lr = 1/np.sqrt(2*(np.sqrt(individual.n_values)))
        lr_prime = 1/(np.sqrt(2*individual.n_values))
        beta = math.pi/36
        normal_matr_prime = np.random.normal(0,lr_prime,1)

        for sigma in range(individual.n_values):

            # Update our sigmas
            normal_matr = np.random.normal(0,lr,1)
            individual.sigmas[sigma] = individual.sigmas[sigma]*(
                        np.exp(normal_matr+normal_matr_prime))

            # Update angles
            alphas_noise = np.random.normal(0,beta,len(individual.alphas))
            individual.alphas = individual.alphas + alphas_noise

            # Check something, i dunno remember why tho
            individual.alphas[individual.alphas > math.pi] = individual.alphas[individual.alphas > math.pi] - 2*math.pi*np.sign(individual.alphas[individual.alphas > math.pi])

            #Calculate C matrix
            count = 0
            C = np.identity(individual.n_values)
            for i in range(individual.n_values-1):
                for j in range(i+1,individual.n_values):
                    R = np.identity(individual.n_values)
                    R[i,i] = math.cos(individual.alphas[count])
                    R[j,j] = math.cos(individual.alphas[count])
                    R[i,j] = -math.sin(individual.alphas[count])
                    R[j,i] = math.sin(individual.alphas[count])
                    C = np.dot(C, R)
                    count += 1
            s = np.identity(individual.n_values)
            np.fill_diagonal(s, individual.sigmas)
            C = np.dot(C, s)
            C = np.dot(C, C.T)

            # Update offspring
            sigma_std = np.random.multivariate_normal(mean=np.full((individual.n_values),fill_value=0), cov=C)
            fix = np.array([ random.gauss(0,i) for i in sigma_std ])
            individual.values =  individual.values + fix