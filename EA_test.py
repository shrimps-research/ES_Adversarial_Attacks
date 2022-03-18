import numpy as np
from classes.Individual import *
from classes.Population import *
from classes.Recombination import *
from classes.Mutation import *
from classes.Selection import *
from classes.EA import *

from classes.Evaluation import *

import numpy as np

# Hyperparameter optimisation
evaluation_function = Ackley().evaluate #eval_fun
is_minimization = True
budget = 50000
parent_size = 35
offspring_size = 320
values_size = 150
recombination = Intermediate(offspring_size)
mutation = IndividualSigma()
selection = OneCommaL()
fallback_patience = 1000000 #budget/10
verbose = 1



ea = EA(evaluation_function=evaluation_function,
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