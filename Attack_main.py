from classes.Individual import *
from classes.Population import *
from classes.Recombination import *
from classes.Mutation import *
from classes.Selection import *
from classes.EA import *

from classes.Evaluation import *

import numpy as np
import argparse

def main():
    
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-func', action='store', 
                        dest='eval_func', type=str,
                        default='ackley')
    parser.add_argument('-min', action='store', 
                        dest='is_min', type=bool,
                        default=True)
    parser.add_argument('-b', action='store', 
                        dest='budget', type=int,
                        default=50000)
    parser.add_argument('-ps', action='store', 
                        dest='parent_size', type=int,
                        default=20)
    parser.add_argument('-os', action='store', 
                        dest='offspring_size', type=int,
                        default=140)
    parser.add_argument('-vs', action='store', 
                        dest='value_size', type=int,
                        default=100)
    parser.add_argument('-r', action='store', 
                        dest='recombination', type=str,
                        default='intermediate')
    parser.add_argument('-m', action='store', 
                        dest='mutation', type=str,
                        default='individual_sigma')
    parser.add_argument('-s', action='store', 
                        dest='selection', type=str,
                        default='one_comma_l')
    parser.add_argument('-fp', action='store', 
                        dest='fallback_patience', type=int,
                        default=1000000)
    parser.add_argument('-v', action='store', 
                        dest='verbose', type=int,
                        default=1)
    args = parser.parse_args()
    if args.verbose:
        print("arguments passed:",args)
    
    # Dictionaries to keep all our Classes
    recombinations = {  'intermediate': Intermediate(args.offspring_size) }

    mutations = {       'individual_sigma': IndividualSigma(),
                        'correlated': Correlated() }

    selections = {      'one_plus_l': OnePlusL(),
                        'one_comma_l': OneCommaL() }

    eval_funs = {       'ackley': Ackley().evaluate,
                        'rastringin': Rastringin.evaluate }

    # Create evolutionary Algorithm
    ea = EA(evaluation_function=eval_funs[args.eval_func],
            is_minimization=args.is_min,
            budget=args.budget,
            parent_size=args.parent_size,
            offspring_size=args.offspring_size,
            values_size=args.value_size,
            recombination=recombinations[args.recombination],
            mutation=mutations[args.mutation],
            selection=selections[args.selection],
            fallback_patience=args.fallback_patience,
            verbose=args.verbose)
            

    best_individual, best_eval = ea.run()
    print(f"best eval: {best_eval}")

if __name__ == "__main__":
    main()