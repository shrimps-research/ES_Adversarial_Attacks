import sys
import os
import argparse
from PIL import Image
import skimage
import time

# Setting paths to folders
sys.path.append('..')
sys.path.append('../ES_adversarial_attacks/')

from ES_adversarial_attacks.Population import *
from ES_adversarial_attacks.Recombination import *
from ES_adversarial_attacks.Mutation import *
from ES_adversarial_attacks.Selection import *
from ES_adversarial_attacks.DNN_Models import *
from ES_adversarial_attacks.Evaluation import *
from ES_adversarial_attacks.EA import *

def main():
    # command line arguments
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
    parser.add_argument('-r', action='store', 
                        dest='recombination', type=str,
                        default=None)
    parser.add_argument('-m', action='store', 
                        dest='mutation', type=str,
                        default='individual_sigma')
    parser.add_argument('-s', action='store', 
                        dest='selection', type=str,
                        default='one_comma_l')
    parser.add_argument('-e', action='store', 
                        dest='epsilon', type=float,
                        default=0.05)
    parser.add_argument('-fp', action='store', 
                        dest='fallback_patience', type=int,
                        default=None)
    parser.add_argument('-sn', action='store',
                        dest='start_noise', type=str,
                        default=None)
    parser.add_argument('-v', action='store', 
                        dest='verbose', type=int,
                        default=1)
    args = parser.parse_args()
    if args.verbose:
        print("arguments passed:",args)

    # dictionaries to keep all our Classes
    recombinations = {  'intermediate': Intermediate(),
                        'discrete': Discrete(),
                        None: None }

    mutations = {       'individual': IndividualSigma(),
                        'one_fifth': OneFifth(),
                        'one_fifth_alt': OneFifth(alt=True) }

    selections = {      'plus_selection': PlusSelection(),
                        'comma_selection': CommaSelection() }

    models = {          'mnist_classifier' : MnistClassifier,
                        'flower_classifier': FlowerClassifier,
                        'xception_classifier': XceptionClassifier,
                        'vit_classifier': ViTClassifier,
                        'perceiver_classifier': PerceiverClassifier }

    evaluations = {     'ackley': Ackley(),
                        'rastringin': Rastringin(),
                        'crossentropy': 
                                Crossentropy( models[args.model](),
                                                        args.true_label,
                                                        minimize=args.minimize,
                                                        targeted=args.targeted),
                        'crossentropy_similarity': 
                                CrossentropySimilarity( models[args.model](),
                                                        args.true_label,
                                                        minimize=args.minimize,
                                                        targeted=args.targeted) }

    # load original image
    og_img_batch = []
    for img_name in os.listdir(args.input_path):
        img = Image.open(args.input_path + img_name)
        img = np.array(img) / 255.0
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        og_img_batch.append(img)
    og_img_batch = np.stack(og_img_batch)

    # load starting noise
    # TODO fix for batches
    if args.start_noise is None:
        start_noise = None
    else:
        start_noise = Image.open(args.start_noise)
        start_noise = np.array(start_noise) / 255.0
        if len(start_noise.shape) == 2:
            start_noise = np.expand_dims(img, axis=2)
        start_noise -= img
        start_noise = [skimage.measure.block_reduce((i+1)*start_noise[:,:,i], (2,2), np.max) for i in range(start_noise.shape[-1])]
        start_noise = np.dstack(start_noise)

    # Create evolutionary Algorithm
    ea = EA(input_=og_img_batch,
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
    
    start_time = time.time()
    parents, best_indiv, best_eval = ea.run()
    end_time = time.time()
    es_run_time = np.round(end_time - start_time, 2)
    print(f'Total es run time: {es_run_time}')

    # reshape and upsample the best noise
    noise = parents.reshape_ind(best_indiv)
    noise = parents.upsample_ind(noise)

    # save best noise
    Image.fromarray((noise * 255).astype(np.uint8)).save('../results/tench/noise.png')

    # save final image
    noisy_batch = (noise + og_img_batch).clip(0, 1)
    for i, img in enumerate(og_img_batch):
        noisy_img = (img + noise).clip(0, 1)
        if noisy_img.shape[-1] == 1:
            noisy_img = np.squeeze(noisy_img, axis=2)
        noisy_img = (noisy_img * 255).astype(np.uint8)
        Image.fromarray(noisy_img).save(f'../results/tench/noisy_input_{i}.png')

    # predict images
    model = models[args.model]()
    noise_preds = model(noisy_batch).numpy()
    noise_acc = (noise_preds.argmax(axis=1)==args.true_label).size/noise_preds.shape[0]
    normal_preds = model(og_img_batch).numpy()
    normal_acc = (normal_preds.argmax(axis=1)==args.true_label).size/normal_preds.shape[0]

    # print results
    print(f"Best function evaluation: {round(best_eval, 2)}")
    print(f'Original prediction: {normal_acc} on class {args.true_label}')
    print(f'Noised prediction: {noise_acc} on class {args.true_label}')

if __name__ == "__main__":
    main()