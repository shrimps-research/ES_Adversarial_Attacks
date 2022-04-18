import numpy as np
from utilities import nn_interpolate


class Population:
    """ Attributes:
            - pop_size : size of population
            - input_ : path specifing the location of the input image
            - epsilon : maximum possible noise as a value between [0,1]
            - mutation : required in order to control alpha initialization based on mutation
            - downsample : value between [0,1] that scales the size of the image
    """
    def __init__(self, input_, pop_size, mutation, epsilon, downsample, start_noise):
        self.epsilon = epsilon
        self.downsample = downsample
        self.mutation = mutation
        self.input_ = input_
        self.pop_size = pop_size
        if downsample is None:
            self.ind_dim = input_.size
        else:  # TODO generalize to generic data (only img now) -> maybe add ImagePopulation class
            self.ind_side_len = int(input_.shape[0] * downsample)
            self.ind_dim = self.ind_side_len * self.ind_side_len * self.input_.shape[-1]
        # initialize individuals
        if start_noise is None:
            self.individuals = np.random.uniform(0, 1, size=(self.pop_size, self.ind_dim))
        else:
            self.individuals = start_noise.reshape(1, -1).repeat(self.pop_size, axis=0)
        # initialize sigmas
        self.init_sigmas()
        # self.alphas = np.deg2rad(np.random.uniform(0,360, size=(pop_size, int((ind_dim*(ind_dim-1))/2))))

    def init_sigmas(self):
        """ Initialize sigma values depending on the mutation method of choice.
        """
        if self.mutation.__class__.__name__ == "OneSigma":
            self.sigmas = np.random.uniform(max(0, np.min(self.individuals)/6), np.max(self.individuals)/6, size=self.pop_size)
        else:
            self.sigmas = np.random.uniform(max(0, np.min(self.individuals)/6), np.max(self.individuals)/6, size=(self.pop_size, self.ind_dim))

    def reshape_ind(self, individual):
        """ Reshape a single individual
        """
        if self.downsample is None:
            return individual.reshape(self.input_.shape)
        else:   # TODO generalize to generic data (only img now) -> maybe add ImagePopulation class
            return individual.reshape((self.ind_side_len, self.ind_side_len, self.input_.shape[2]))

    def upsample_ind(self, individual):
        """ Upsample individual back to the original size of the image by interpolation.
        """
        return np.dstack([nn_interpolate(individual[:,:,i], self.input_.shape[:2]) for i in range(self.input_.shape[2])])

    def upsample_general(self,individual, shape):
        """ Upsample individual to size of the image by interpolation.
        """
        return np.dstack([nn_interpolate(individual[:,:,i], shape[:2]) for i in range(shape[2])])
        
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

    def evaluate_(self, evaluation):
        """ Deprecated. Use evaluate instead.
            Evaluate the whole batch with single individual forward passes
        """
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

    def evaluate(self, evaluation):
        """ Evaluate the fitness of the whole batch in a single forward pass
        """
        if self.epsilon is not None:
            self.individuals = self.individuals.clip(-self.epsilon, self.epsilon)
        batch = []
        for individual in self.individuals:
            # reshape noise (individual) to input shape
            individual = self.reshape_ind(individual)
            # upscale to input shape
            if self.downsample is not None:  # TODO generalize to generic data (only img now) -> maybe add ImagePopulation class
                individual = self.upsample_ind(individual)
            # add original input and append to batch list
            batch.append(individual + self.input_)
        # stack batch as BxHxWxC
        batch = np.stack(batch)
        # evaluate batch
        batch_evals = evaluation(batch)
        self.fitnesses = list(batch_evals)