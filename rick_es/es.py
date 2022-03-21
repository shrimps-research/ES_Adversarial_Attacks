import numpy as np
import random

class EvolutionaryStrategy():
    def __init__(self, input_, predict, target, budget, num_pop, num_off, individual_sigma, comma_selection, recombination, epsilon=None, targeted=False, downsample=None):
        self.input_ = input_
        self.downsample = downsample
        if downsample is None:
            self.input_dim = input_.size
        else:
            side_length = int(input_.shape[0] * downsample)
            self.input_dim = side_length * side_length * input_.shape[2]
        self.predict = predict
        self.epsilon = epsilon
        self.target = target
        self.targeted = targeted
        self.init_budget = budget
        self.budget = budget
        self.num_off = num_off
        self.individual_sigma = individual_sigma
        # generate parents population
        if individual_sigma:
            self.init_pop_individual(num_pop)
            self.mutate = self.mutate_individual
        else:
            self.init_pop_onesigma(num_pop)
            self.mutate = self.mutate_onesigma
        # select selection type
        if comma_selection:
            self.selection = self.selection_comma
        else:
            self.selection = self.selection_plus
        # select recombination type
        if recombination == "discrete":
            self.recombination = self.recombination_discrete
        elif recombination == "global_discrete":
            self.recombination = self.recombination_global_discrete
        elif recombination == "intermediate":
            self.recombination = self.recombination_intermediate
        else:
            self.recombination = None

    def reshape_ind(self, individual):
        if self.downsample is None:
            return individual.reshape(self.input_.shape)
        else:
            side_len = np.rint((self.input_dim/self.input_.shape[2])**(1/2)).astype(np.uint32)
            return individual.reshape((side_len, side_len, self.input_.shape[2]))

    def init_pop_onesigma(self, num_pop):
        self.x = np.random.uniform(0, 1, size=(num_pop, self.input_dim))
        self.sigma = np.random.uniform(np.min(self.x)/6, np.max(self.x)/6, size=num_pop)

    def init_pop_individual(self, num_pop):
        self.x = np.random.uniform(0, 1, size=(num_pop, self.input_dim))
        self.sigma = np.random.uniform(np.min(self.x)/6, np.max(self.x)/6, size=(num_pop, self.input_dim))

    def recombination_discrete(self):
        self.x = []
        self.sigma = []
        for _ in range(self.num_off):
            # get pair
            i1, i2 = random.sample(range(len(self.parents)), k=2)
            x1, x2 = self.parents[i1], self.parents[i2]
            s1, s2 = self.sigma_parents[i1], self.sigma_parents[i2]
            # probabilities of selecting each component (and/or sigma) of x1
            prob_x1 = np.random.uniform(0, 1, size=self.input_dim)
            # recombinate components and sigmas
            self.x.append(np.array([x1_i if prob_x1_i >= 0.5 else x2_i for x1_i, x2_i, prob_x1_i in zip(x1, x2, prob_x1)]))
            if self.individual_sigma:
                # select each sigma identically to the components selection
                self.sigma.append(np.array([s1_i if prob_s1_i >= 0.5 else s2_i for s1_i, s2_i, prob_s1_i in zip(s1, s2, prob_x1)]))
            else:
                # select at random one of the two sigmas
                self.sigma.append(s1 if np.random.uniform(0, 1) >= 0.5 else s2)
        self.x = np.vstack(self.x)
        if self.individual_sigma:
            self.sigma = np.vstack(self.sigma)
        else:
            self.sigma = np.array(self.sigma)

    def recombination_global_discrete(self):
        self.x = []
        self.sigma = []
        for _ in range(self.num_off):
            # for each component draw from a uniform [0,1)
            # divide [0,1) in num parents intervals and pick the component of the parent in that interval
            prob = np.random.uniform(0, 1, size=self.input_dim)
            prob = (prob * self.parents.shape[0]).astype("int")
            # recombinate components and sigmas
            self.x.append(np.array([self.parents[p, i] for i, p in enumerate(prob)]))
            if self.individual_sigma:
                # select each sigma identically to the components selection
                self.sigma.append(np.array([self.sigma_parents[p, i] for i, p in enumerate(prob)]))
            else:
                # select at random one of the two sigmas
                self.sigma.append(self.sigma_parents[int(np.random.uniform(0, 1) * self.parents.shape[0])])
        self.x = np.vstack(self.x)
        if self.individual_sigma:
            self.sigma = np.vstack(self.sigma)
        else:
            self.sigma = np.array(self.sigma)  

    def recombination_intermediate(self):
        self.x = []
        self.sigma = []
        for _ in range(self.num_off):
            # get pair
            i1, i2 = random.sample(range(len(self.parents)), k=2)
            x1, x2 = self.parents[i1], self.parents[i2]
            s1, s2 = self.sigma_parents[i1], self.sigma_parents[i2]
            # recombinate components and sigmas
            self.x.append(np.array([(x1_i + x2_i) / 2 for x1_i, x2_i in zip(x1, x2)]))
            if self.individual_sigma:
                # select the sigma of each component as the mean of the sigmas of the two components
                self.sigma.append(np.array([(s1_i + s2_i) / 2 for s1_i, s2_i in zip(s1, s2)]))
            else:
                # sigma is the mean of the two sigmas
                self.sigma.append((s1 + s2) / 2)
        self.x = np.vstack(self.x)
        if self.individual_sigma:
            self.sigma = np.vstack(self.sigma)
        else:
            self.sigma = np.array(self.sigma)

    def mutate_onesigma(self):
        # transform sigmas
        tau_0 = 1 / np.sqrt(self.input_dim)
        self.sigma = self.sigma * np.exp(np.random.normal(0, tau_0, size=self.sigma.shape[0]))
        # mutate components
        variations = np.stack([np.random.normal(0, s, size=self.input_dim) for s in self.sigma])
        self.x += variations

    def mutate_individual(self):
        # transform sigmas
        tau = 1 / np.sqrt(2 * np.sqrt(self.input_dim))
        tau_prime = 1 / np.sqrt(2 * self.input_dim)
        # one draw from N(0, tau') per individual
        tau_prime_drawns = np.random.normal(0, tau_prime, size=self.sigma.shape[0]).reshape(-1, 1).repeat(self.input_dim, axis=1)
        # one draw from N(0, tau) per sigma (individuals x components)
        tau_drawns = np.random.normal(0, tau, size=self.sigma.shape)
        self.sigma = self.sigma * np.exp(tau_drawns + tau_prime_drawns)
        # mutate components
        variations = np.random.normal(0, self.sigma)
        self.x += variations

    def selection_comma(self, _, offsprings_fitness):
        # select the n fittest offsprings, where n is the number of parent individuals
        sorted_ind = np.argsort(offsprings_fitness)
        self.x = self.x[sorted_ind][:self.parents.shape[0]]
        self.sigma = self.sigma[sorted_ind][:self.parents.shape[0]]

    def selection_plus(self, parents_fitness, offsprings_fitness):
        # select the n fittest individuals between parents+offsprings, where n is the number of parent individuals
        sorted_ind = np.argsort(np.hstack([parents_fitness, offsprings_fitness]))
        self.x = np.vstack([self.parents, self.x])[sorted_ind][:self.parents.shape[0]]
        if self.individual_sigma:
            self.sigma = np.vstack([self.sigma_parents, self.sigma])[sorted_ind][:self.parents.shape[0]]
        else:
            self.sigma = np.hstack([self.sigma_parents, self.sigma])[sorted_ind][:self.parents.shape[0]]

    def evaluate(self, individual):
        """ if targeted attack, use crossentropy (-log(pred)) on target
            if untargeted attack, use negative crossentropy (log(pred)) on target
        """
        # reshape noise (individual) to input shape
        individual = self.reshape_ind(individual)
        # feasible input space contraint
        # if np.min(input_) < 0 or np.max(input_) > 1:
        #     return np.inf if self.targeted else 0
        # prediction
        predictions = self.predict(individual, self.input_)
        # return loss
        return (-1 if self.targeted else 1) * np.log(predictions[self.target])

    def evaluate_population(self, verbose):
        budget_left = self.budget - self.x.shape[0]
        if budget_left >= 0:
            self.budget = budget_left
            evaluations = [self.evaluate(x) for x in self.x]
        else:
            self.budget = 0
            evaluations = [self.evaluate(x) for x in self.x[:budget_left]]
        # check if there is a better solution than the current one
        best_eval_ind = np.argmin(evaluations)
        if evaluations[best_eval_ind] < self.best_f:
            self.best_f = evaluations[best_eval_ind]
            self.best_x = self.x[best_eval_ind]
            if verbose:
                print(f"[{self.init_budget-self.budget}/{self.init_budget}] Current best value: {self.best_f}")
        return evaluations

    def run(self, verbose=False):
        print("Optimum value:", int(not self.targeted) if self.targeted else -np.inf)
        if self.targeted:
            self.best_f = np.inf
        else:
            self.best_f = 0
        self.best_x = None
        while self.budget > 0:
            # clip parents
            if self.epsilon is not None:
                self.x = self.x.clip(-self.epsilon, self.epsilon)
            # evaluate parents
            parents_fitness = self.evaluate_population(verbose)
            if self.budget == 0: break  # can't evaluate the offsprings
            # store parents
            self.parents = self.x.copy()
            self.sigma_parents = self.sigma.copy()
            # generate offsprings
            if self.recombination is not None:
                self.recombination()
            # mutate offsprings
            self.mutate()
            # clip offsprings
            if self.epsilon is not None:
                self.x = self.x.clip(-self.epsilon, self.epsilon)
            # evaluate offsprings
            offsprings_fitness = self.evaluate_population(verbose)
            # selection between parents and offsprings
            self.selection(parents_fitness, offsprings_fitness)
        print(f"Run finished with best value {self.best_f} and used budget of {self.init_budget-self.budget}")