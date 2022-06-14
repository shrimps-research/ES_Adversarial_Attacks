import numpy as np
import random
import math
from ES_adversarial_attacks.Population import Population

class Mutation:
    def mutate(self):
        """ Mutate the whole population
        """
        pass

    def __call__(self, *args):
        self.mutate(*args)


class IndividualSigma(Mutation):
    """ Individual sigma method.
    """
    def mutate(self, population: Population, *_):
        tau = 1 / np.sqrt(2 * np.sqrt(population.individuals.shape[1]))
        tau_prime = 1 / np.sqrt(2 * population.individuals.shape[1])
        # one draw from N(0, tau') per individual
        tau_prime_drawns = np.random.normal(0, tau_prime, size=population.sigmas.shape[0])
        tau_prime_drawns = tau_prime_drawns.reshape(-1, 1).repeat(population.sigmas.shape[1], axis=1)
        # one draw from N(0, tau) per sigma (individuals x components)
        tau_drawns = np.random.normal(0, tau, size=population.sigmas.shape)
        # mutate sigmas
        population.sigmas = population.sigmas * np.exp(tau_drawns + tau_prime_drawns)
        # mutate components
        variations = np.random.normal(0, population.sigmas)
        population.individuals += variations


# TODO add support for one-sigma
class OneFifth(Mutation):
    """ 1/5 success rule method.
    """
    def __init__(self, alt=False):
        if alt:
            self.mutate = self.mutate_alt
        else:
            self.mutate = self.mutate_

    def mutate_(self, population: Population, gen_succ: int, gen_tot: int, *_):
        c = 0.95
        k = 40  # sigmas reset patience
        # reset sigmas
        if gen_tot % k == 0:
            population.init_sigmas()
        # increase sigmas (explore more)
        elif gen_succ/gen_tot > 0.20:
            population.sigmas /= c
        # decrease sigmas (exploit more)
        elif gen_succ/gen_tot < 0.20:
            population.sigmas *= c
        # mutate components
        variations = np.random.normal(0, population.sigmas)
        population.individuals += variations

    def mutate_alt(self, population: Population, gen_succ: int, gen_tot: int, *_):
        """ Individual sigma + OneFifth
        """
        tau = 1 / np.sqrt(2 * np.sqrt(population.individuals.shape[1]))
        tau_prime = 1 / np.sqrt(2 * population.individuals.shape[1])
        # one draw from N(0, tau') per individual
        tau_prime_drawns = np.random.normal(0, tau_prime, size=population.sigmas.shape[0])
        tau_prime_drawns = tau_prime_drawns.reshape(-1, 1).repeat(population.sigmas.shape[1], axis=1)
        # one draw from N(0, tau) per sigma (individuals x components)
        tau_drawns = np.random.normal(0, tau, size=population.sigmas.shape)
        # mutate sigmas
        population.sigmas = population.sigmas * np.exp(tau_drawns + tau_prime_drawns)
        # scale sigmas and mutate individuals
        self.mutate_(population, gen_succ, gen_tot)