import numpy as np


class Individual:
    def __init__(self, n_values, initial_fitness=np.inf):
        self.n_values = n_values
        self.values = np.random.uniform(-1,1,size=n_values)
        #self.sigmas = np.random.uniform(0,1,size=n_values)
        self.sigmas = np.random.uniform(np.min(self.n_values)/6,np.max(self.n_values)/6,size=n_values)
        self.alphas = np.deg2rad(np.random.uniform(0,360, int((n_values*(n_values-1))/2)))
        self.fitness = initial_fitness

    def copy_individual(self):
        i = Individual(self.n_values, self.fitness)
        i.values = self.values
        i.sigmas = self.sigmas
        i.alphas = self.alphas
        i.fitness = self.fitness
        return i