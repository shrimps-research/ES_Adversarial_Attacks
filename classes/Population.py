from classes.Individual import Individual
import numpy as np

class Population:

    def __init__(self, size, n_values, is_min):
        self.size = size
        self.is_min = is_min
        if is_min:
            self.individuals = [Individual(n_values, np.inf) for i in range(size)]
        else:
            self.individuals = [Individual(n_values, np.NINF) for i in range(size)]

    def ave_fitness(self):
        return sum(individual.fitness for individual in self.individuals) / len(self.individuals)

    def max_fitness(self, get_index=False):
        if get_index:
            return np.argmax([ individual.fitness for individual in self.individuals])
        return max(individual.fitness for individual in self.individuals)

    def min_fitness(self,get_index=False):
        if get_index:
            return np.argmin([ individual.fitness for individual in self.individuals])
        return min(individual.fitness for individual in self.individuals)
        

    def best_fitness(self, get_index=False):
        if self.is_min:
            if get_index:
                return self.min_fitness(get_index=True)
            return self.min_fitness()
        else:
            if get_index:
                return self.max_fitness(get_index=True)
            return self.max_fitness()

    def best_individual(self):
        return self.individuals[self.best_fitness(get_index=True)]