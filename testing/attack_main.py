import sys
import argparse
from PIL import Image
from numpy import uint8

# Setting paths to folders
sys.path.append('..')
sys.path.append('../classes/')

from classes.Population import *
from classes.Recombination import *
from classes.Mutation import *
from classes.Selection import *
from classes.DNN_Models import *
from classes.Evaluation import *
from classes.EA import *

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
    parser.add_argument('-in', action='store',
                        dest='input_path', type=str,
                        default='../data/img_data/five.png')
    parser.add_argument('-tl', action='store',
                        dest='true_label',
                        default=0, type=int)
    parser.add_argument('-ps', action='store', 
                        dest='parent_size', type=int,
                        default=20)
    parser.add_argument('-os', action='store', 
                        dest='offspring_size', type=int,
                        default=140)
    parser.add_argument('-vs', action='store', 
                        dest='value_size', type=int,
                        default=None)
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

    selections = {      'plus_selection': PlusSelection(),
                        'comma_selection': CommaSelection() }

    models = {          'mnist_classifier' : MnistClassifier,
                        'flower_classifier': FlowerClassifier,
                        'xception_classifier': XceptionClassifier }

    eval_funs = {       'ackley': Ackley().evaluate,
                        'rastringin': Rastringin().evaluate,
                        'classification_crossentropy': 
                                ClassifierCrossentropy( models[args.model](),
                                                        args.true_label,
                                                        targeted=False  # TODO args
                                                      ).evaluate }

    # Load original image
    original_img = Image.open(args.input_path)
    original_img = np.array(original_img) / 255.0
    if args.value_size is None:
        args.value_size = original_img.size
    if len(original_img.shape) == 2:
        original_img = np.expand_dims(original_img, axis=2)
    # Create evolutionary Algorithm
    ea = EA(input_=original_img,
            evaluation_function=eval_funs[args.eval_func],
            minimization=args.is_min,
            budget=args.budget,
            parents_size=args.parent_size,
            offspring_size=args.offspring_size,
            values_size=args.value_size,
            recombination=recombinations[args.recombination],
            mutation=mutations[args.mutation],
            selection=selections[args.selection],
            fallback_patience=args.fallback_patience,
            verbose=args.verbose,
            downsample=False)  # TODO args
            
    parents, best_index = ea.run()

    # Save noisy image
    noise = parents.reshape_ind(parents.individuals[best_index])
    noise = parents.upsample_ind(noise)
    # clip image + noise in [0,1] then subtract input to obtain clipped noise
    noise = (original_img + noise).clip(0, 1) - original_img
    if noise.shape[2] == 1:
        noise = np.squeeze(noise, axis=2)
    Image.fromarray((noise * 255).astype(np.uint8)).save('../results/noise.png')
    

    # Save final image
    if original_img.shape[2] == 1:
        original_img = np.squeeze(original_img, axis=2)
    noisy_input = ((original_img + noise) * 255).astype(np.uint8)
    Image.fromarray(noisy_input).save('../results/noisy_input.png')

    # Predict images
    model = models[args.model]().model
    noise_preds = model(np.expand_dims(noise+original_img, axis=0))
    normal_preds = model(np.expand_dims(np.zeros(original_img.shape)+original_img, axis=0))

    # Print results
    print(f"Best function evaluation: {parents.fitnesses[best_index]}")
    print(f'Initial prediction: {np.argmax(normal_preds)} - Confidence: {np.max(normal_preds)}')
    print(f'Noised prediction: {np.argmax(noise_preds)} - Confidence: {np.max(noise_preds)}')

if __name__ == "__main__":
    main()