import math
import numpy as np
from torch import chunk
from tqdm import tqdm
from typing import Tuple
from collections import deque
import Evaluation
from Population import Population
from scipy.linalg import eigh
import time

def iterate_efficiently(input, output, chunk_size):
  # create an empty array to hold each chunk
  # the size of this array will determine the amount of RAM usage
  # holder = np.zeros([chunk_size,800,800], dtype='uint16')

  # iterate through the input, replace with ones, and write to output
  for i in range(input.shape[0]):
    # output[i]
      if i % chunk_size == 0:
          up_to = min(i+chunk_size, input.shape[0]-1)
          # holder[:] = input[i:i+chunk_size] # read in chunk from input
          # holder += 5 # perform some operation
          # output[i:i+chunk_size] = holder # write chunk to output
          output[i:up_to] = (input[i:up_to] + input.T[i:up_to])/2 # write chunk to output

class CMA_ES():
  def __init__(
    self,
    alpha,
    beta,
    model,
    input_,
    epsilon,
    downsample,
    evaluation: Evaluation

  ):
    self.evaluation = evaluation
    self.model = model
    self._alpha = alpha
    self._beta = beta
    self.device = None
    self._mu = np.random.uniform(alpha, beta) 
    self._sigma =  0.3 * (beta-alpha)
    # del self.population.sigmas
    self.trend = deque([], 100)
    if downsample is None:
      n_dim = input_[0, :].size
    else:  # TODO generalize to generic data (only img now) -> maybe add ImagePopulation class
      ind_side_len = int(input_.shape[1] * downsample)
      n_dim = ind_side_len * ind_side_len * input_.shape[-1]

    # print(n_dim)
    # exit()
    # TODO: Check pop size again
    # self._pop_size = 4 + math.floor(3*math.log(n_dim))
    self._pop_size = 32
    self.population = Population(input_=input_, pop_size=self._pop_size, mutation=None, start_noise=None, epsilon=epsilon, downsample=downsample)
    n_dim = self.population.ind_dim
    self._n_dim = n_dim
    print(self._pop_size)
    print(n_dim)
    # exit()
    # f = np.memmap('memmapped.dat', dtype=np.float32,
    #           mode='w+', shape=(n_dim, n_dim))
    # self._C = np.identity(n_dim)
    np.save('_C.npy', np.identity(n_dim))
    self._C = np.load('_C.npy', mmap_mode='r+')
    # print(self._C)
    self._pc = 0
    self._psigma = 0
    self.g = 0 # generation counter
    self._mean = np.full((n_dim), self._mu)

    self._EPS = 1e-8
    self._SIGMA_THRESHOLD = 1e32
    print('EigenDecomposition starting')
    self._eigen_decomposition()
    print('EigenDecomposition finished')
    self._init_weights()
    print('Weight initialization finished')
    # self._B = None
    # self._D = None


  def _init_weights(self):
    mu = math.floor(self._pop_size/2)

    # equation 49 from paper
    weights_prime = np.array(
        [
            math.log((self._pop_size + 1) / 2) - math.log(i)
            for i in range(1, self._pop_size + 1)
        ]
    )
    # weights_prime up to mu value 
    mu_eff = (np.sum(weights_prime[:mu])**2) / np.sum(weights_prime[:mu]**2)
    # weights prime from mu till end
    mu_eff_minus = (np.sum(weights_prime[mu:])**2) / np.sum(weights_prime[mu:]**2)

    alpha_cov = 2

    # Equration 56 from paper 
    cc = (4 + mu_eff/self._n_dim)/(self._n_dim + 4 + 2*mu_eff/self._n_dim)
    # Equation 57 from paper
    c1 = alpha_cov/((self._n_dim + 1.3) ** 2 + mu_eff)
    # Equation 58 from paper
    c_mu = min(1 - c1,
              alpha_cov * ((mu_eff - 2 + 1/mu_eff)/
                ((self._n_dim + 2)**2) + alpha_cov*mu_eff/2)
            )
    # Equation 50 from paper
    alpha_mu_minus =  1 + c1/c_mu  
    # Equation 51 from paper
    alpha_mu_eff_minus = 1 + (2*mu_eff_minus)/(mu_eff+2) 
    # Equation 52 from paper
    alpha_pos_def_minus = (1 -c1 - c_mu)/(self._n_dim*c_mu)  

    # Calculate minimum alpha (part of weights)
    min_alpha = min(alpha_mu_minus, alpha_mu_eff_minus, alpha_pos_def_minus)

    # Equation 53 from paper 
    positive_weights_sum = 1/(np.sum(weights_prime[weights_prime>0]))
    negative_weights_sum = min_alpha/(np.sum(weights_prime[weights_prime<0]))
    weights = np.where( weights_prime >= 0,
                        positive_weights_sum * weights_prime,
                        negative_weights_sum * weights_prime
                      )

    # Equation 54 from paper
    cm = 1

    # Equation 55 from paper
    c_sigma = (mu_eff + 2)/(self._n_dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff-1)/self._n_dim+1)-1) + c_sigma
    # Save to object
    self._mu = mu
    self._mu_eff = mu_eff
    self._cc = cc
    self._c1 = c1
    self._c_mu = c_mu
    self._c_sigma = c_sigma
    self._d_sigma = d_sigma
    self._cm = cm
    self._weights = weights
    # evolution paths
    self._p_sigma = np.zeros(self._n_dim)
    self._pc = np.zeros(self._n_dim)
    self._B, self._D = None, None




  def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
    input = np.memmap('_C.npy', dtype='float32', shape=(self._n_dim, self._n_dim), mode='w+')

    # create a memmap array to store the output
    output = np.memmap('_C.npy', dtype='float32', shape=(self._n_dim, self._n_dim), mode='w+')
    iterate_efficiently(input,output,1000)
    print("Iteration Finished")
    # self._C = output
    np.save('_C.npy', output)
    print("Saving Finished")
    start = time.time()
    print("Numpy")
    D2, B = np.linalg.eigh(output)
    # D2, B = np.linalg.eigh(self._C)
    end = time.time()
    print(end - start)

    # start = time.time()
    # print("Scipy")
    # D2, B = eigh(self._C)
    # end = time.time()
    # print(end - start)

    # exit()

    D = np.sqrt(np.where(D2 < 0, self._EPS, D2))
    self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)
    self._B, self._D = B, D
    np.save('_C.npy', self._C)
    del input
    del output

  def _getBD(self):
    return self._B, self._D

  def sample_pop(self):
    # self._eigen_decomposition()
    # ~ N(m, σ^2 C)
    pop = np.array([np.random.multivariate_normal(mean=self._mean, cov=(self._sigma**2 * self._C)) 
              for i in range(self._pop_size)])
    return pop

  def cma_procedure(self):
    solutions = self.population.individuals
    fitnesses = self.population.fitnesses
    self.g += 1
    solutions_sorted = [solutions[i[0]] for i in sorted(enumerate(fitnesses), key=lambda x:x[1])]
    # ___________ Sample new population ___________
    self._eigen_decomposition()
    B, D = self._getBD()
    self._B, self._D = None, None


    x_k = np.array([s for s in solutions_sorted])  
    y_k = (x_k - self._mean) / self._sigma  

    # ___________ Selection and recombination ___________
    # Equtation 41 from paper
    yw = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  
    self._mean += self._cm * self._sigma * yw

    # ___________ Step size control ___________
    # C^(-1/2) = B D^(-1) B^T
    # Note: D can be inverted by inverting its diagonal elements
    eps = 1e-64

    C_1_2 = np.array(B.dot(np.diag(1/(D+eps))).dot(B.T))
    # Equation 43 from paper
    self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
                      self._c_sigma * (2 - self._c_sigma) * self._mu_eff
                    ) * C_1_2.dot(yw)

    # E||N(0, I)|| From Paper this can be approximated as follows (p.28 of paper)
    e = math.sqrt(self._n_dim) * (
        1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim ** 2))
    )

    # Equation 44 from paper
    self._sigma *= np.exp(
        (self._c_sigma / self._d_sigma) * (np.linalg.norm(self._p_sigma) / e - 1)
    )
    # TODO: check later if needed
    self._sigma = min(self._sigma, self._SIGMA_THRESHOLD)


    # ___________ Covariance Matrix Adaptation ___________
    # Look at page 28 of paper for how to calculate h_sigma 
    left_condition = np.linalg.norm(self._p_sigma) /\
                              math.sqrt(
                                1 - (1 - self._c_sigma)**(2*(self.g+1))
                              )
    right_condition = (1.4 + 2/(self._n_dim+1)) * e
    h_sigma = 1.0 if left_condition < right_condition else 0.0
    # Equation 45 from paper
    self._pc = (1-self._cc)*self._pc + h_sigma * math.sqrt(self._cc*(2-self._cc)*self._mu_eff)*yw
    # Equation 46 from paper
    w_i_zero = self._weights * np.where(
            self._weights >= 0,
            1,
            self._n_dim / (np.linalg.norm(C_1_2.dot(y_k.T), axis=0) ** 2 + self._EPS),
        )
    # Equation 47 from paper
    # calculate δ(hσ) from page 28
    delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc) 
    assert delta_h_sigma <= 1
    rank_mu = np.sum(
            np.array([w * np.outer(y, y).transpose() for w, y in zip(w_i_zero, y_k)]), axis=0
        )
    self._C = (1 + self._c1*delta_h_sigma - self._c1 - self._c_mu*np.sum(self._weights))*self._C \
      +  self._c1 * np.outer(self._pc,self._pc).transpose() + self._c_mu * rank_mu

  def run(self, max_epochs=5):
    for gen in tqdm(range(max_epochs)):
      print(self._C, self._mean)
      self.population.individuals = self.sample_pop()
      self.population.evaluate(self.evaluation.evaluate)
      self.cma_procedure()
      print(f'Gen {gen}: {self.population.fitnesses}')
      np.save('best_indiv.npy', np.array(self._mean))
    # best_eval, best_index = self.population.parents.best_fitness(self.minimize)
    # best_indiv = self.population.parents.individuals[best_index]
    # return self.population.individuals, self._mean, self.population.best_fitness(True)
    self.population.individuals = np.array([self._mean])
    self.population.pop_size = 1
    self.population.evaluate(self.evaluation.evaluate)
    return self.population, np.array(self._mean), self.population.fitnesses[0]