import sys
import argparse
from PIL import Image
import skimage

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
    parser.add_argument('-eval', action='store', 
                        dest='evaluation', type=str,
                        default='ackley')
    parser.add_argument('-min', action='store_true', 
                        dest='minimize')
    parser.add_argument('-t', action='store_true', 
                        dest='targeted')
    parser.add_argument('-ds', action='store',
                        dest='downsample', type=float,
                        default=None)
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
    parser.add_argument('-e', action='store', 
                        dest='epsilon', type=float,
                        default=0.05)
    parser.add_argument('-r', action='store', 
                        dest='recombination', type=str,
                        default=None)
    parser.add_argument('-m', action='store', 
                        dest='mutation', type=str,
                        default='individual_sigma')
    parser.add_argument('-s', action='store', 
                        dest='selection', type=str,
                        default='one_comma_l')
    parser.add_argument('-fp', action='store', 
                        dest='fallback_patience', type=int,
                        default=1000000)
    parser.add_argument('-sn', action='store',
                        dest='start_noise', type=str,
                        default=None)
    parser.add_argument('-v', action='store', 
                        dest='verbose', type=int,
                        default=1)
    args = parser.parse_args()
    if args.verbose:
        print("arguments passed:",args)

    # Dictionaries to keep all our Classes
    recombinations = {  'intermediate': Intermediate(args.offspring_size),
                        None: None }

    mutations = {       'individual': IndividualSigma(),
                        'one_fifth': OneFifth(),
                        'one_fifth_alt': OneFifth(alt=True),
                        'correlated': Correlated() }

    selections = {      'plus_selection': PlusSelection(),
                        'comma_selection': CommaSelection() }

    models = {          'mnist_classifier' : MnistClassifier,
                        'flower_classifier': FlowerClassifier,
                        'xception_classifier': XceptionClassifier,
                        'vit_classifier': ViTClassifier,
                        'perceiver_classifier': PerceiverClassifier }

    evaluations = {     'ackley': Ackley(),
                        'rastringin': Rastringin(),
                        'classification_crossentropy': 
                                ClassifierCrossentropy( models[args.model](),
                                                        args.true_label,
                                                        minimize=args.minimize,
                                                        targeted=args.targeted) }

    # Load original image
    original_img = Image.open(args.input_path)
    if args.model != "mnist_classifier":
        original_img = original_img.convert("RGB")
    original_img = np.array(original_img) / 255.0
    if len(original_img.shape) == 2:
        original_img = np.expand_dims(original_img, axis=2)

    # Load starting noise
    if args.start_noise is None:
        start_noise = None
    else:
        start_noise = Image.open(args.start_noise)
        start_noise = np.array(start_noise) / 255.0
        if len(start_noise.shape) == 2:
            start_noise = np.expand_dims(original_img, axis=2)
        start_noise -=  - original_img
        start_noise = [skimage.measure.block_reduce((i+1)*start_noise[:,:,i], (2,2), np.max) for i in range(start_noise.shape[2])]
        start_noise = np.dstack(start_noise)

    # Create evolutionary Algorithm
    ea = EA(input_=original_img,
            evaluation=evaluations[args.evaluation],
            minimize=args.minimize,
            budget=args.budget,
            parents_size=args.parent_size,
            offspring_size=args.offspring_size,
            recombination=recombinations[args.recombination],
            mutation=mutations[args.mutation],
            selection=selections[args.selection],
            fallback_patience=args.fallback_patience,
            verbose=args.verbose,
            epsilon=args.epsilon,
            downsample=args.downsample,
            start_noise=start_noise)
            
    parents, best_indiv, best_eval = ea.run()

    # Save noisy image
    noise = parents.reshape_ind(best_indiv)
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
    model = models[args.model]()
    noise_preds = model(np.expand_dims(noise+original_img, axis=0))
    normal_preds = model(np.expand_dims(np.zeros(original_img.shape)+original_img, axis=0))

    # Print results
    print(noise_preds.shape)
    print(f"Best function evaluation: {round(best_eval)}")
    print(f'Original prediction: {np.max(normal_preds)} on class {np.argmax(normal_preds)}')
    print(f'Noised prediction: {np.max(noise_preds)} on class {np.argmax(noise_preds)}')
    print(f'Noised prediction: {noise_preds[0, args.true_label]} on original class {args.true_label}')

if __name__ == "__main__":
    main()