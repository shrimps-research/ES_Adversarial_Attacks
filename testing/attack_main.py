import os
import sys
import time
import argparse
import skimage
from PIL import Image
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from transformers import add_start_docstrings

# Setting paths to folders
sys.path.append('..')
sys.path.append('../ES_adversarial_attacks/')

from ES_adversarial_attacks.Population import *
from ES_adversarial_attacks.Recombination import *
from ES_adversarial_attacks.Mutation import *
from ES_adversarial_attacks.Selection import *
from ES_adversarial_attacks.DNN_Models import *
from ES_adversarial_attacks.Evaluation import *
from ES_adversarial_attacks.ES import *

def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataloader', action='store_true', 
                        dest='dataloader')
    parser.add_argument('-batches', type=int, nargs='+',
                        default=(1000,256))
    parser.add_argument('-eval', action='store', 
                        dest='evaluation', type=str,
                        default='ackley')
    parser.add_argument('-min', action='store_true', 
                        dest='minimize')
    parser.add_argument('-t', action='store_true', 
                        dest='targeted')
    parser.add_argument('-d', action='store',
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
    parser.add_argument('-device', action='store', 
                        dest='device', type=str,
                        default="cpu")
    args = parser.parse_args()
    if args.verbose:
        print("arguments passed:",args)

    # define cuda device if specified
    if args.device is not None:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print("running on", device)

    # dictionaries to keep all our Classes
    recombinations = {  'intermediate': Intermediate(),
                        'discrete': Discrete(),
                        'global_discrete': GlobalDiscrete(),
                        None: None }

    mutations = {       'individual': IndividualSigma(),
                        'one_fifth': OneFifth(),
                        'one_fifth_alt': OneFifth(alt=True) }

    selections = {      'plus_selection': PlusSelection(),
                        'comma_selection': CommaSelection() }

    models = {          'vgg_classifier': VGGClassifier,
                        'xception_torch': XceptionTorch,
                        'xception_classifier': XceptionClassifier,
                        'vit_classifier': ViTClassifier,
                        'perceiver_classifier': PerceiverClassifier }
    model = models[args.model]()
    model.model = model.model.to(device)
    model.model.eval()

    evaluations = {     'ackley': Ackley(),
                        'rastringin': Rastringin(),
                        'crossentropy': 
                                Crossentropy(
                                    model,
                                    args.true_label,
                                    device=device,
                                    minibatch=args.batches[1],
                                    minimize=args.minimize,
                                    targeted=args.targeted),
                        'blind_evaluation': 
                                BlindEvaluation(
                                    model,
                                    args.true_label,
                                    device=device,
                                    minibatch=args.batches[1],
                                    minimize=args.minimize,
                                    targeted=args.targeted) }

    # load original image
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        lambda x: torch.permute(x, (1, 2, 0))
    ])
    # used for stock pytorch models
    new_transforms = T.Compose([
        model.transforms, # stock transforms
        # permutation required for the Population class...
        lambda x: torch.permute(x, (1, 2, 0))
    ])
    if args.dataloader:
        if args.model == "vgg_classifier" or args.model == "xception_torch":
            og_data = datasets.ImageFolder(args.input_path, transform=new_transforms) # model.transforms does not work for some reason
        else:
            og_data = datasets.ImageFolder(args.input_path, transform=transform)
        og_data = DataLoader(og_data, batch_size=args.batches[0], shuffle=True)
        # initial images predictions
        normal_acc = 0
        for batch in og_data:
            batch = batch[0].numpy()
            normal_preds = model(batch, device).cpu().numpy()
            normal_acc += (normal_preds.argmax(axis=1)==args.true_label).sum()
        normal_acc /= len(og_data.dataset)
        print(f'\nOriginal prediction: {normal_acc*100} on original class {args.true_label}')
    else:
        og_data = []
        for img_name in os.listdir(args.input_path):
            img = Image.open(args.input_path + img_name)
            if args.model == "vgg_classifier" or args.model == "xception_torch":
                img = new_transforms(img)
            else:
                img = np.array(transform(img))
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            og_data.append(img)
        og_data = np.stack(og_data)
        # initial image prediction
        normal_preds = model(og_data, device).cpu().numpy()
        normal_acc = (normal_preds.argmax(axis=1)==args.true_label).sum()/normal_preds.shape[0]
        confidence = normal_preds.max(axis=1)[0]
        print(f'\nOriginal prediction: {np.round(confidence*100,2)} confidence on class {normal_preds.argmax(axis=1).item()}, orig class: {args.true_label}')

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
    es = ES(input_=og_data,
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
    
    # run ES 
    start_time = time.time()
    parents, best_indiv, best_eval = es.run()
    end_time = time.time()
    es_run_time = np.round(end_time - start_time, 2)
    print(f'Total es run time: {es_run_time}')

    # reshape and upsample the best noise
    noise = parents.reshape_ind(best_indiv)
    if args.downsample is not None:
        noise = parents.upsample_ind(noise)

    # save best noise
    Image.fromarray((noise * 255).astype(np.uint8)).save('../results/noise.png')

    # save final image
    if og_data.__class__.__name__ != "DataLoader":
        noisy_batch = (noise + og_data).clip(0, 1)
        for i, img in enumerate(og_data):
            noisy_img = (img + noise).clip(0, 1)
            if noisy_img.shape[-1] == 1:
                noisy_img = np.squeeze(noisy_img, axis=2)
            noisy_img = (noisy_img * 255).astype(np.uint8)
            Image.fromarray(noisy_img).save(f'../results/noisy_input_{i}.png')
            Image.fromarray(np.uint8(img*255)).save(f'../results/cropped_input_{i}.png')

    # predict images
    if og_data.__class__.__name__ != "DataLoader":
        noise_preds = model(noisy_batch, device).cpu().numpy()
        noise_acc = (noise_preds.argmax(axis=1)==args.true_label).sum()/noise_preds.shape[0]
        normal_preds = model(og_data, device).cpu().numpy()
        normal_acc = (normal_preds.argmax(axis=1)==args.true_label).sum()/normal_preds.shape[0]
    else:
        noise_acc = 0
        normal_acc = 0
        for batch in og_data:
            batch = batch[0].numpy()
            noisy_batch = (noise + batch).clip(0, 1)
            noise_preds = model(noisy_batch, device).cpu().numpy()
            noise_acc += (noise_preds.argmax(axis=1)==args.true_label).sum()
            normal_preds = model(batch, device).cpu().numpy()
            normal_acc += (normal_preds.argmax(axis=1)==args.true_label).sum()
        noise_acc /= len(og_data.dataset)
        normal_acc /= len(og_data.dataset)

    # print results
    print(f"Best function evaluation: {round(best_eval, 2)}")
    print(f'[Original] correct predictions on class {args.true_label}: {normal_acc*100}%')
    print(f'[Noised] correct predictions on class {args.true_label}: {noise_acc*100}%')

if __name__ == "__main__":
    main()