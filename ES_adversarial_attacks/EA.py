from ES_adversarial_attacks.Population import Population
import numpy as np


class EA:
    """ Main Evolutionary Strategy class
    """
    def __init__(self, input_, evaluation, minimize, budget,
                parents_size, offspring_size,
                recombination, mutation, selection,
                fallback_patience, verbose, epsilon=0.05, downsample=None,
                one_fifth=False, start_noise=None) -> None:
        self.evaluation = evaluation
        self.minimize = minimize
        self.budget = budget
        self.parents_size = parents_size
        self.offspring_size = offspring_size
        self.recombination = recombination
        self.mutation = mutation
        self.selection = selection
        self.one_fifth = one_fifth
        self.fallback_patience = fallback_patience
        self.verbose=verbose
        one_sigma = True if mutation.__class__.__name__ == "OneSigma" else False
        self.parents = Population(input_, self.parents_size, one_sigma, epsilon, downsample, start_noise)
        self.offspring = Population(input_, self.offspring_size, one_sigma, epsilon, downsample, start_noise)

    def run(self):
        """ Main function to run the Evolutionary Strategy
        """
        # Initialize budget and best evaluation (as worst possible)
        curr_budget = 0
        curr_patience = 0
        best_eval = self.evaluation.worst_eval()

        # Initialize (generation-wise) success probability params
        # Success means finding a new best individual in a given gen. of offspring
        # gen_tot=num. of offspring gen., gen_succ=num. of successfull gen.
        gen_tot = 0
        gen_succ = 0

        # Initial parents evaluation step
        self.parents.evaluate(self.evaluation.evaluate)
        best_eval, best_index = self.parents.best_fitness(self.minimize)
        best_indiv = self.parents.individuals[best_index]
        curr_budget += self.parents_size

        while curr_budget < self.budget:
            gen_tot += 1

            # Recombination: creates new offspring
            if self.recombination is not None and self.parents_size > 1:
                self.recombination(self.parents, self.offspring)
            
            # Mutation: mutate individuals (offspring)
            self.mutation(self.offspring, gen_succ, gen_tot)

            # Evaluate offspring population
            self.offspring.evaluate(self.evaluation.evaluate)
            curr_budget += self.offspring_size
            curr_patience += self.offspring_size  # TODO patience

            # Next generation parents selection
            self.selection(self.parents, self.offspring, self.minimize)

            # Update the best individual in case of success
            curr_best_eval= self.parents.fitnesses[0]
            success = False
            if self.minimize:
                if curr_best_eval < best_eval:
                    success = True
            else:
                if curr_best_eval > best_eval:
                    success = True
            if success:
                gen_succ += 1
                best_indiv = self.parents.individuals[0]
                best_eval = curr_best_eval
                curr_patience = 0  # Reset patience since we found a new best
                if self.verbose > 1:
                    print(f"[{curr_budget}/{self.budget}] New best eval: {round(best_eval, 2)}" + \
                    f" | Pred: {round(np.abs(np.exp(best_eval)),2)} | P_succ: {round(gen_succ/gen_tot, 2)}")

        return self.parents, best_indiv, best_eval