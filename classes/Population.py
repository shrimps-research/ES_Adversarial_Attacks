from classes.Individual import Individual
import numpy as np


class Population:
    """
    size: size of population
    n_values: size of value array of individual

    individual's fitness is initialized to np.inf .
    """

    def __init__(self, size, n_values):

        self.size = size
        self.n_values = n_values
        self.individuals = [Individual(n_values, np.inf) for i in range(size)]


    def all_fitnesses(self):
        """
        Return an array of the population's fitnesses.
        """
        return np.array([individual.fitness for individual in self.individuals])


    def ave_fitness(self):
        """
        Calculates the average fitness of the population.
        """
        return sum(self.all_fitnesses()) / len(self.individuals)


    def max_fitness(self, get_index=False):
        """
        Calculates the maximum fitness of the population.
        """
        if get_index:
            return np.argmax(self.all_fitnesses())
        return np.max(self.all_fitnesses())


    def min_fitness(self,get_index=False):
        """
        Calculates the minimum fitness of the population.
        """
        if get_index:
            return np.argmin(self.all_fitnesses())
        return np.min(self.all_fitnesses())
        

    def best_fitness(self, is_min=True, get_index=False):
        """
        Calculates the mbest fitness based on the problem.

        is_min: True if it is a minimisation problem
        get_index: returns the index of the best fitted individual
        """
        if is_min:
            if get_index:
                return self.min_fitness(get_index=True)
            return self.min_fitness()
        else:
            if get_index:
                return self.max_fitness(get_index=True)
            return self.max_fitness()


    def best_individual(self):
        """
        Returns the best individual in the population.
        """
        return self.individuals[self.best_fitness(get_index=True)]


    def evaluate_fitness(self,evaluation_function):
        for individual in self.individuals:
            individual.fitness = evaluation_function(individual.values)
