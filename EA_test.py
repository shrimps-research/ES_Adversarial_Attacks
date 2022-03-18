import numpy as np
from classes.Individual import *
from classes.Population import *
from classes.Recombination import *
from classes.Mutation import *
from classes.Selection import *
from classes.EA import *

import numpy as np


def eval_fun(values):
    return sum(values)

is_minimization = True
budget = 1000
parent_size = 10
offspring_size = 60
values_size = 200
recombination = Intermediate(offspring_size)
mutation = CustomSigma()
selection = OnePlusL()
fallback_patience = 10000
verbose = 2



ea = EA(evaluatin_function=eval_fun,
        is_minimization=is_minimization,
        budget=budget,
        parent_size=parent_size,
        offspring_size=offspring_size,
        values_size=values_size,
        recombination=recombination,
        mutation=mutation,
        selection=selection,
        fallback_patience=fallback_patience,
        verbose=verbose)

best_individual, best_eval = ea.run()
print(f"best eval: {best_eval}")