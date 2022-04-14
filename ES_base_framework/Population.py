import numpy as np

class Population:
    """ Attributes:
            - pop_size : size of population
            - input_ : path specifing the location of the input image
            - mutation : defines the mutation to be used in order to initialize parameters
    """
    def __init__(self, pop_size, ind_dim, mutation):
        self.mutation = mutation
        self.pop_size = pop_size
        self.ind_dim = ind_dim

        # initialize individual values
        self.individuals = np.random.uniform(0, 1, size=(self.pop_size, self.ind_dim))
        # initialize fitnesses
        self.fitnesses = []
        # initialize sigmas
        self.init_sigmas()
        # initialize alphas if necessary
        if self.mutation.__class__.__name__ == "Correlated":
            self.alphas = np.deg2rad(np.random.uniform(0,360, size=(self.pop_size, int((self.ind_dim*(self.ind_dim-1))/2))))
        
  
    def init_sigmas(self):
        """ Initialize sigma values depending on the mutation method of choice.
        """
        if self.mutation.__class__.__name__ == "OneSigma":
            self.sigmas = np.random.uniform(max(0, 
                                                np.min(self.individuals)/6), 
                                                np.max(self.individuals)/6, 
                                                size=self.pop_size)
        else:
            self.sigmas = np.random.uniform(max(0, 
                                                np.min(self.individuals)/6), 
                                                np.max(self.individuals)/6, 
                                                size=(self.pop_size, self.ind_dim))

        
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
        """ Evaluate the current population
        """
        self.fitnesses = [evaluation(ind) for ind in self.individuals]
