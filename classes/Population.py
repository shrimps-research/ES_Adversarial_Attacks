import numpy as np
from utilities import nn_interpolate


class Population:
    """ size: size of population
        n_values: size of value array of individual

        individual's fitness is initialized to np.inf
    """
    def __init__(self, input_, pop_size, one_sigma, epsilon, downsample):
        self.epsilon = epsilon
        self.downsample = downsample
        self.one_sigma = one_sigma
        self.input_ = input_
        self.pop_size = pop_size
        if downsample is None:
            ind_dim = input_.size
        else:  # TODO generalize to generic data (only img now) -> maybe add ImagePopulation class
            self.ind_side_len = int(input_.shape[0] * downsample)
            ind_dim = self.ind_side_len * self.ind_side_len * self.input_.shape[-1]
        # initialize individuals
        self.individuals = np.random.uniform(0, 1, size=(pop_size, ind_dim))
        # initialize sigmas
        if one_sigma:
            self.sigmas = np.random.uniform(np.min(self.individuals)/6, np.max(self.individuals)/6, size=pop_size)
        else:
            self.sigmas = np.random.uniform(np.min(self.individuals)/6, np.max(self.individuals)/6, size=(pop_size, ind_dim))
        # self.alphas = np.deg2rad(np.random.uniform(0,360, size=(pop_size, int((ind_dim*(ind_dim-1))/2))))

    def reshape_ind(self, individual):
        """ reshape a single individual
        """
        if self.downsample is None:
            return individual.reshape(self.input_.shape)
        else:   # TODO generalize to generic data (only img now) -> maybe add ImagePopulation class
            return individual.reshape((self.ind_side_len, self.ind_side_len, self.input_.shape[2]))

    def upsample_ind(self, individual):
        return np.dstack([nn_interpolate(individual[:,:,i], self.input_.shape[:2]) for i in range(self.input_.shape[2])])

    def max_fitness(self):
        """ Calculates the maximum fitness of the population.
            Return max fitness and its index
        """
        return np.max(self.fitnesses), np.argmax(self.fitnesses)

    def min_fitness(self):
        """ Calculates the minimum fitness of the population.
            Return min fitness and its index
        """
        return np.min(self.fitnesses), np.argmin(self.fitnesses)
        
    def best_fitness(self, minimize=True):
        """ Calculates the best fitness based on the problem.
            Return the best its fitness and index

            minimize: True if it is a minimization problem, False if maximization
        """
        if minimize:
            best_fitness, best_index = self.min_fitness()
        else:
            best_fitness, best_index = self.max_fitness()
        return best_fitness, best_index

    def evaluate(self, evaluation):
        self.fitnesses = []
        if self.epsilon is not None:
            self.individuals = self.individuals.clip(-self.epsilon, self.epsilon)
        for individual in self.individuals:
            # reshape noise (individual) to input shape
            individual = self.reshape_ind(individual)
            # upscale to input shape
            if self.downsample is not None:  # TODO generalize to generic data (only img now) -> maybe add ImagePopulation class
                individual = self.upsample_ind(individual)
            # evaluate input + noiose
            self.fitnesses.append(evaluation(individual, self.input_))