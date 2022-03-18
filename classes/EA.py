from classes.Individual import Individual
from classes.Population import Population
from classes.Recombination import Intermediate
from classes.Mutation import CustomSigma
from classes.Selection import OneCommaL, OnePlusL

import numpy as np

class EA:

    def __init__(self,evaluation_function, is_minimization, budget,
                parent_size, offspring_size, values_size,
                recombination, mutation, selection,
                fallback_patience, verbose) -> None:

        self.evaluation_function = evaluation_function
        self.is_minimization = is_minimization
        self.budget = budget
        self.parent_size = parent_size
        self.offspring_size = offspring_size
        self.values_size = values_size
        self.recombination = recombination
        self.mutation = mutation
        self.selection = selection
        self.fallback_patience = fallback_patience
        self.verbose=verbose
        self.parent = Population(self.parent_size, self.values_size)
        self.offspring = Population(self.offspring_size, self.values_size)

    def run(self):
        if not self.is_minimization:
            print(f'check if minimization is ok')
            exit()

        # Initialize budget and best evaluation
        curr_budget = 0
        curr_patience = 0
        best_eval = np.inf if self.is_minimization else np.NINF

        # Initial evaluation step
        self.parent.evaluate_fitness(self.evaluation_function)
        curr_budget += self.parent_size

        while curr_budget < self.budget:

            # Temporary minimization handler
            if self.is_minimization:

                # Keep track of best fit individual and evaluation
                curr_best_eval = self.parent.best_fitness(is_min=True)
                if curr_best_eval < best_eval:
                    best_eval = curr_best_eval
                    best_individual = self.parent.individuals[self.parent.best_fitness(True,True)]
                    # Reset patience since we found a new best
                    curr_patience = 0
                    if self.verbose > 1:
                        print(f"new best val: {best_eval}, used budget: {curr_budget}")

                # Recombination: creates new offspring
                offspring = self.recombination.recombine(self.parent)

                # Mutation: mutate all individuals
                self.mutation.mutate_population(offspring)

                # Evaluate offspring population
                offspring.evaluate_fitness(self.evaluation_function)
                curr_budget += self.offspring_size
                curr_patience += self.offspring_size

                # Next generation parents selection with fallback
                if curr_patience >= self.fallback_patience:
                    self.parents = OneCommaL().select(self.parent,offspring)
                    curr_patience = 0
                    if self.verbose:
                        print(f'+++ Fallback activated on budget: {curr_budget}')
                else:
                    self.parents = self.selection.select(self.parent, offspring)

        return best_individual, best_eval

                