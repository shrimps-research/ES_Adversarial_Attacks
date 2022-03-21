from classes.Population import Population
import numpy as np


class EA:
    """ Current constraints: 
        - offspring population must be at least as big as parent population
        - maximisation not completely implemented 
    """
    def __init__(self, input_, evaluation, minimize, budget,
                parents_size, offspring_size,
                recombination, mutation, selection,
                fallback_patience, verbose, epsilon=0.05, downsample=None) -> None:
        self.evaluation = evaluation
        self.minimize = minimize
        self.budget = budget
        self.parents_size = parents_size
        self.offspring_size = offspring_size
        self.recombination = recombination
        self.mutation = mutation
        self.selection = selection
        self.fallback_patience = fallback_patience
        self.verbose=verbose
        one_sigma = False if mutation.__class__.__name__ == "IndividualSigma" else True
        self.parents = Population(input_, self.parents_size, one_sigma, epsilon, downsample)
        self.offspring = Population(input_, self.offspring_size, one_sigma, epsilon, downsample)

    def run(self):
        """ Main function to run the Evolutionary Strategy
        """
        # Initialize budget and best evaluation (as worst possible)
        curr_budget = 0
        curr_patience = 0
        best_eval = self.evaluation.worst_eval()

        # Initial evaluation step
        self.parents.evaluate(self.evaluation.evaluate)
        best_eval, best_index = self.parents.best_fitness(self.minimize)
        curr_budget += self.parents_size

        while curr_budget < self.budget:
            # Recombination: creates new offspring
            self.recombination(self.parents, self.offspring)
            
            # Mutation: mutate all individuals
            self.mutation(self.offspring)

            # Evaluate offspring population
            self.offspring.evaluate(self.evaluation.evaluate)
            curr_budget += self.offspring_size
            curr_patience += self.offspring_size  # TODO patience

            # Next generation parents selection with fallback
            self.selection(self.parents, self.offspring)

            # Evaluate parent population
            self.parents.evaluate(self.evaluation.evaluate)
            curr_budget += self.parents_size

            # Keep track of best fit individual and evaluation
            curr_best_eval, curr_best_index = self.parents.best_fitness(self.minimize)
            # TODO do this minimiz/maximiz part better
            if self.minimize:
                if curr_best_eval < best_eval:
                    best_eval = curr_best_eval
                    best_index = curr_best_index
                    curr_patience = 0  # Reset patience since we found a new best
                    if self.verbose > 1:
                        print(f"[{curr_budget}/{self.budget}] New best value: {best_eval}")
            else:
                if curr_best_eval > best_eval:
                    best_eval = curr_best_eval
                    best_index = curr_best_index
                    if self.verbose > 1:
                        print(f"[{curr_budget}/{self.budget}] New best value: {best_eval}")

        return self.parents, best_index