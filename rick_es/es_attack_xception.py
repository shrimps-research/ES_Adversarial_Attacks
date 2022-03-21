from PIL import Image
import numpy as np
import tensorflow as tf
from utilities import nn_interpolate
from es import EvolutionaryStrategy

def predict(noise, input_):
    """ evaluate input_+noise using the chosen model
    """
    input_ = (input_ + noise).reshape((1, *input_.shape))
    return model(input_)[0].numpy()

def upsample_predict(noise, input_):
    """ upsample some noise to the target shape (input_.shape)
        then predict input_+noise using the chosen model
    """
    upsampled_noise = np.dstack([nn_interpolate(noise[:,:,i], input_.shape[:2]) for i in range(noise.shape[2])])
    return predict(upsampled_noise, input_)

##### CONFIGURATION #####
downsample = 0.06
epsilon = 0.1
targeted = False
budget = 100
num_pop = 2
num_off = 2
individual_sigma = True
comma_selection = False
recombination = "discrete"
print(f"CONFIGURATION: parents:{num_pop}, offsprings:{num_off}, individual:{individual_sigma}, comma:{comma_selection}, recombination:{recombination}")

##### DATASET #####
input_ = Image.open("imagenet_samples/tench.png")
input_ = np.asarray(input_, dtype=np.uint8) / 255.0
target = 0  # 0:tench, 963:pizza

##### MODEL TO ATTACK #####
model = tf.keras.applications.Xception(weights='imagenet', include_top=True, input_shape=input_.shape)

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
image = ((input_ + noise) * 255).astype(np.uint8)
# save attack image
Image.fromarray(image).save("results/tench_noisy.png")