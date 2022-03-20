from classes.Population import Population
import numpy as np


class EA:
    """ Current constraints: 
        - offspring population must be at least as big as parent population
        - maximisation not yet implemented 
    """
    def __init__(self, input_, evaluation_function, minimization, budget,
                parents_size, offspring_size, values_size,
                recombination, mutation, selection,
                fallback_patience, verbose, downsample=False) -> None:
        self.evaluation_function = evaluation_function
        self.minimization = minimization
        self.budget = budget
        self.parents_size = parents_size
        self.offspring_size = offspring_size
        self.values_size = values_size
        self.recombination = recombination
        self.mutation = mutation
        self.selection = selection
        self.fallback_patience = fallback_patience
        self.verbose=verbose
        self.downsample = downsample
        self.parents = Population(input_, self.parents_size, self.values_size, downsample)
        self.offspring = Population(input_, self.offspring_size, self.values_size, downsample)

    def run(self):
        """ Main function to run the Evolutionary Strategy
        """
        # Initialize budget and best evaluation
        curr_budget = 0
        curr_patience = 0
        best_eval = np.inf if self.minimization else -np.inf

        # Initial evaluation step
        self.parents.evaluate(self.evaluation_function)
        curr_budget += self.parents_size

        best_eval, best_index = self.parents.best_fitness(self.minimization)

        while curr_budget < self.budget:
            # Temporary minimization handler
            if self.minimization:
                # Keep track of best fit individual and evaluation
                curr_best_eval, curr_best_index = self.parents.best_fitness(self.minimization)
                if curr_best_eval < best_eval:
                    best_eval = curr_best_eval
                    best_index = curr_best_index
                    
                    curr_patience = 0  # Reset patience since we found a new best
                    if self.verbose > 1:
                        print(f"new best val: {best_eval}, used budget: {curr_budget}")
                
                # Recombination: creates new offspring
                self.recombination(self.parents, self.offspring)
                
                # Mutation: mutate all individuals
                self.mutation(self.offspring)

                # Evaluate offspring population
                self.offspring.evaluate(self.evaluation_function)

                # Increase used budget and control variables
                curr_budget += self.offspring_size
                curr_patience += self.offspring_size
                if (self.verbose == 1) and (curr_budget%1000 == 0):
                    print(f"current best {best_eval}, current budget: {curr_budget}/{self.budget}")

                # Next generation parents selection with fallback
                self.selection(self.parents, self.offspring)

        return self.parents, best_index