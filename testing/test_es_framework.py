import sys
import time
# Setting paths to folders
sys.path.append('..')
sys.path.append('../ES_base_framework/')

from ES_base_framework.Population import *
from ES_base_framework.Recombination import *
from ES_base_framework.Mutation import *
from ES_base_framework.Selection import *
from ES_base_framework.Evaluation import *
from ES_base_framework.EA import *

def main():
    #random.seed(0)
    #np.random.seed(0)

    recomb = None
    mutation = OneFifth()
    selection = PlusSelection()
    evaluation = Ackley()

    repetitions = 100

    ea = EA(minimize=True,
            budget=10000,
            parents_size=6,
            offspring_size=36,
            individual_size=100,
            recombination=recomb,
            mutation=mutation,
            selection=selection,
            evaluation=evaluation,
            verbose=0)

    best_evals = []
    best_budgets = []
    start_time = time.time()
    for _ in range(repetitions):
        _, best_eval, best_budget = ea.run()
        best_evals.append(best_eval)
        best_budgets.append(best_budget)
    end_time = time.time()
    print(f"Run time: {np.round(end_time - start_time, 3)}")
    print(f"mean best eval: {np.round(np.mean(best_evals),4)}, mean budget: {np.mean(best_budgets)}, in {repetitions} repetitions")

if __name__ == "__main__":
    main()