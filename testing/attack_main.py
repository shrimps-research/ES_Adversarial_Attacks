from pyexpat import model
import sys
from numpy import uint8

# Setting paths to folders
sys.path.append('..')
sys.path.append('../classes/')
from classes.Individual import *
from classes.Population import *

from classes.Recombination import *
from classes.Mutation import *
from classes.Selection import *

from classes.DNN_Models import *
from classes.Evaluation import *
from classes.EA import *

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
    parser.add_argument('-model', action='store',
                        dest='model', type=str,
                        default='mnist_classifier')
    parser.add_argument('-img', action='store',
                        dest='img', type=str,
                        default='../data/img_data/zero.png')
    parser.add_argument('-img_class', action='store',
                        dest='img_class',
                        default=0)
    parser.add_argument('-ps', action='store', 
                        dest='parent_size', type=int,
                        default=20)
    parser.add_argument('-os', action='store', 
                        dest='offspring_size', type=int,
                        default=140)
    parser.add_argument('-vs', action='store', 
                        dest='value_size', type=int,
                        default=100)
    parser.add_argument('-e', action='store', 
                        dest='epsilon', type=float,
                        default=0.05)
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

    models = {          'mnist_classifier' : MnistClassifier(),
                        'flower_classifier': FlowerClassifier() }

    eval_funs = {       'ackley': Ackley().evaluate,
                        'rastringin': Rastringin().evaluate,
                        'classification_crossentropy': 
                                ClassifierCrossentropy( models[args.model], 
                                                        args.img, 
                                                        args.img_class,
                                                        epsilon=args.epsilon
                                                      ).evaluate
                }

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


    # Save original image
    original_img = PIL.Image.open(args.img)
    original_img.save('output/original.png')
    original_img_arr = np.array(original_img)

    # Save noisy image
    noise_arr = best_individual.values
    noise_arr = np.array(noise_arr*255,dtype=np.uint8).clip(0,255).reshape(original_img_arr.shape)
    noise = PIL.Image.fromarray(noise_arr)
    noise.save('output/noise.png')

    # Save final image
    combined_arr = original_img_arr + noise_arr*args.epsilon
    combined_arr = combined_arr.clip(0,255).astype(uint8)
    combined_img = PIL.Image.fromarray(combined_arr)
    combined_img.save('output/final.png')

    # Predict images
    evaluator = ClassifierCrossentropy(models[args.model], args.img, args.img_class)
    normal_preds = evaluator.predict(original_img_arr)
    noise_preds = evaluator.predict(combined_arr)

    # Print results
    print(f"best function evaluation: {best_eval}")
    print(f'initial prediction: {np.argmax(normal_preds[0])} confidence: {np.max(normal_preds[0])}')
    print(f'noised prediction: {np.argmax(noise_preds[0])} confidence: {np.max(noise_preds[0])}')

if __name__ == "__main__":
    main()