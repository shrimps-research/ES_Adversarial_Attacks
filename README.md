# Evolutionary Strategies based Adversarial Attacks (under construction)
This repository contains a framework for applying evolutionary strategies (ES) on arbitrary majorization-minimization (MM) problems. In the specific scope of our project, we applied ES for adversarial attacks on deep neural networks. The experiments conducted are only image oriented. Specifically, we search for otimal *noises*, which, combined to an original image, are able to fool a network (e.g. misclassification).  Given that we deal with very high-dimensional search spaces, we implemented different methods to deal with the curse of dimensionality.


## Authors
<a href="https://github.com/OhGreat">Dimitrios Ierinomakys</a>, <a href="https://github.com/riccardomajellaro">Riccardo Majellaro</a>, <a href="https://github.com/doctorblinch">Ivan Horokhovskyi</a>, <a href="https://github.com/pavlosZakkas">Pavlos Zakkas</a>, <a href="https://www.linkedin.com/in/andreas-paraskeva-2053141a3/">Andreas Paraskeva</a>

## Run ES for Adversarial Attacks
In this section we describe how to launch an ES for adversarial attacks, using the `attack_main.py` script. Using our script as a reference it is possible to create personalized scripts to apply ES on arbitrary problems.

In order to launch the ES execute the following commands from the main directory:
```
cd testing
python attack_main.py [-args]  
```
The following arguments can be used:
- `-eval` : defines the fitness evaluation function. e.g. "crossentropy"
- `-min` : use this flag in case of minimization problem. Maximization problem when not used.
- `-t` : use this flag in case of a targeted attack. Untargeted attack when not used.
- `-ds` : downsample factor in (0,1] where 1 means no downscaling. Lower values mean greater downscaling.
- `-b` : number of maximum fitness function evaluations.
- `-model` : defines the model to be used in the evaluation. e.g. "xception_classifier"
- `-in` : defines the path of the input image used to attack the chosen model.
- `-tl` : ground truth label of the input image.
- `-ps` : defines the size of the parent population. e.g. 2
- `-os` : defines the size of the offspring population. e.g. 4
- `-r` : defines the recombination to use in the ES strategy. eg. "intermediate"
- `-m` : defines the mutation to use in the ES strategy. eg. "one_fifth"
- `-s` : defines selection to use in the ES strategy. eg. "plus_selection"
- `-e` :  defines the epsilon used when clipping the noise. The noise is then constrained in [-e,e]
- `-sn` : to be used if you want to initialize the parent population with a single predefined noise. Set this argument to the path of the noise in the form of a numpy array.
- `-v` : verbose intensity parameter. Set to a value between 0 and 2, with 2 as the most intense.

## Run evaluation
In this section we describe how to launch an evaluation on a chosen model and image (with both noise or not), using the `evaluate.py` script.

In order to launch the evaluation execute the following commands from the main directory:
```
cd testing
python evaluate.py
```

In order to customize the evaluation you need to modify the script. A CLI will probably be provided in the future.


## Results (so far)
### Attack on a simple MLP trained on MNIST
<img src="https://github.com/OhGreat/ES_Adversarial_Attacks/blob/main/images/mnist_example.png" width="700px" />

### Attack on Xception trained on ImageNet (mutation: intermediate)
<img src="https://github.com/OhGreat/ES_Adversarial_Attacks/blob/main/images/xception_ind_example.png" width="700px" />

### Attack on Xception trained on ImageNet (mutation: 1/5 success rule)
<img src="https://github.com/OhGreat/ES_Adversarial_Attacks/blob/main/images/xception_onefifth_example.png" width="700px" />

### Soon: attacks on ViT and Perceiver IO
