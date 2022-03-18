from Individual import Individual
from Population import Population
from Recombination import Intermediate
from Mutation import CustomSigma
from Selection import OneCommaL, OnePlusL

class EA:

    def __init__(self,evaluatin_function, is_minimization, budget,
                parent_size, offspring_size, values_size,
                recombination, mutation, selection,
                fallback_patience, verbose) -> None:

        self.evaluatin_function = evaluatin_function
        self.is_minimization = is_minimization
        self.budget = budget
        self.parent_size = parent_size
        self.offspring_size = offspring_size
        self.values_size = values_size
        self.recombination = recombination
        self.mutation = mutation
        self.selection = selection
        self.fallbacvk_patience = fallback_patience
        self.verbose=verbose

        self.population = Population(self.parent_size, self.values_size)

    def run(self):
        curr_budget = 0
        best_eval = self.population.best_fitness(is_min=True)
        best_individual = self.population.individuals[self.population.best_fitness(True,True)]
        self.population.evaluate_fitness(self.evaluatin_function)
        curr_budget += self.parent_size

        while curr_budget < self.budget:
            if self.is_minimization:
                pass
            pass
        pass
                # Keep track of best fit individual and evaluation
