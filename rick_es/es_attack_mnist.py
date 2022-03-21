from PIL import Image
import numpy as np
import tensorflow as tf
from imgclf_nn import build_DNN
from utilities import nn_interpolate
from es import EvolutionaryStrategy

def predict(noise, input_):
    """ evaluate input_+noise using the chosen model
    """
    input_ = (input_ + noise).reshape((1, *input_.shape))
    return model(input_)[0]

def upsample_predict(noise, input_):
    """ upsample some noise to the target shape (input_.shape)
        then predict input_+noise using the chosen model
    """
    upsampled_noise = np.dstack([nn_interpolate(noise[:,:,i], input_.shape[:2]) for i in range(noise.shape[2])])
    return predict(upsampled_noise, input_)

##### CONFIGURATION #####
downsample = 1
epsilon = 0.05
targeted = False
budget = 1000
num_pop = 2
num_off = 2
individual_sigma = True
comma_selection = True
recombination = "intermediate"
print(f"CONFIGURATION: parents:{num_pop}, offsprings:{num_off}, individual:{individual_sigma}, comma:{comma_selection}, recombination:{recombination}")

##### DATASET #####
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((*x_train.shape, 1))
x_test = x_test.reshape((*x_test.shape, 1))
input_ = x_train[1000]
target = 0

##### MODEL TO ATTACK #####
model = build_DNN()
model.load_weights("imgclf_nn_ckpt/weights")

##### EVOLUTIONARY STRATEGY #####
es = EvolutionaryStrategy(input_, predict if downsample is None else upsample_predict, target, budget, num_pop, num_off, individual_sigma, comma_selection, recombination, epsilon, targeted, downsample)
es.run(verbose=True)
if es.best_x is None:
    print("No fiesible solution found")
    exit()

##### TEST AND SAVE #####
# clip image + noise in [0,1] then subtract input to obtain clipped noise
noise = es.reshape_ind(es.best_x)
if downsample is not None:
    noise = np.dstack([nn_interpolate(noise[:,:,i], input_.shape[:2]) for i in range(input_.shape[2])])
noise = (input_ + noise).clip(0, 1) - input_
# evaluate model with attack image on target label
pred_image_og = predict(np.zeros(input_.shape), input_)[target]
if downsample is None:
    pred_image = predict(noise, input_)[target]
else:
    pred_image = upsample_predict(noise, input_)[target]
print(f"Prediction on target {target}: {pred_image}")
print(f"Original prediction on target {target}: {pred_image_og}")
print(f"Difference of {pred_image_og-pred_image}")
# reproject pixels to [0, 255] and convert them to unsigned int of 1 byte
image = ((input_ + noise) * 255).reshape(input_.shape[:2]).astype(np.uint8)
# save attack image
Image.fromarray(image).save("results/zero_noisy_.png")