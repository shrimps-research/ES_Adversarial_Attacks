# Evolutionary-driven Adversarial Attacks
This repository contains a framework for applying evolutionary strategies (ES) on arbitrary optimization problems. In the specific scope of our project, we applied ES for adversarial attacks on deep neural networks. The experiments conducted are only image oriented. Specifically, we search for otimal *noises*, which, combined to an original image, are able to fool a network (e.g. misclassification).  Given that we deal with very high-dimensional search spaces, we implemented different methodologies to efficiently tackle the problem.

## Authors
<a href="https://github.com/OhGreat">Dimitrios Ieronymakis</a>, <a href="https://github.com/riccardomajellaro">Riccardo Majellaro</a>, <a href="https://github.com/doctorblinch">Ivan Horokhovskyi</a>, <a href="https://github.com/pavlosZakkas">Pavlos Zakkas</a>, <a href="https://www.linkedin.com/in/andreas-paraskeva-2053141a3/">Andreas Paraskeva</a>

## Run ES for Adversarial Attacks
In this section we describe how to launch an ES for adversarial attacks, using the `attack_main.py` script. Using our script as a reference it is possible to create personalized scripts to apply ES on arbitrary problems.

In order to launch the ES execute the following commands from the main directory:
```
cd testing
python attack_main.py [-args]
```
The following arguments can be used:
- `dataloader`: (store_true) will use dataloader to load the images when set to True. This should be used when the dataset of images is huge. If everything can fit into memory, this option is not recommended for better performance.
- `-eval` : (str) defines the fitness evaluation function. e.g. "crossentropy"
- `-min` : (store_true) use this flag in case of minimization problem. Maximization problem when not used.
- `-t` : (store_true) use this flag in case of a targeted attack. Untargeted attack when not used.
- `-d` : (float) downsample factor in (0,1] where 1 means no downscaling. Lower values mean greater downscaling.
- `-b` : (int) number of maximum fitness function evaluations.
- `-model` : (str) defines the model to be used in the evaluation. e.g. "xception_classifier"
- `-in` : (str) defines the path of the input image used to attack the chosen model.
- `-tl` : (int) ground truth label of the input image.
- `-ps` : (int) defines the size of the parent population. e.g. 2
- `-os` : (int) defines the size of the offspring population. e.g. 4
- `-r` : (str) defines the recombination to use in the ES strategy. eg. "intermediate"
- `-m` : (str) defines the mutation to use in the ES strategy. eg. "one_fifth"
- `-s` : (str) defines selection to use in the ES strategy. eg. "plus_selection"
- `-e` :  (float) defines the epsilon used when clipping the noise. The noise is then constrained in [-e,e]
- `-fb` : (int) defines the fallback patience interval. Fallback patience defines the iterations to reset sigmas or population after which we haven't had any improvement.
- `-sn` : (str) to be used if you want to initialize the parent population with a single predefined noise. Set this argument to the path of the noise in the form of a numpy array.
- `-v` : (int) verbose intensity parameter. Set to a value between 0 and 2, with 2 as the most intense.
- `-device` : (str) defines the device to use for model computations. Default is "cpu", but "cuda" ("cuda:0" etc..) can be used for GPU usage.

## Run evaluation
In this section we describe how to launch an evaluation on a chosen model and image (with both noise or not), using the `evaluate.py` script.

In order to launch the evaluation execute the following commands from the main directory:
```
cd testing
python evaluate.py
```

In order to customize the evaluation you need to modify the script. A CLI will probably be provided in the future.


## Examples
Some example scripts to setup configurations are available under the `testing` directory, both in **.sh** and **.bat** formats.

The following image instead, is an example of the results expected with different configurations of epsilon and downsampling, on the Xception classification model.
 <img src="https://github.com/shrimps-research/ES_Adversarial_Attacks/blob/main/images/xception_config_search.png" width="100%" />
