from PIL import Image
import numpy as np
from imgclf_nn import build_DNN

# model to attack
model = build_DNN()
model.load_weights("imgclf_nn_ckpt/weights")

# load attack image and evaluate it on target
target = 5
image = Image.open("results/five_noisy.png")
image = np.asarray(image, dtype=np.uint8) / 255.0
image_og = Image.open("mnist_samples/five.png")
image_og = np.asarray(image_og, dtype=np.uint8) / 255.0
pred_image = model(image.reshape((1, *image.shape, 1)))[0][target]
pred_image_og = model(image_og.reshape((1, *image_og.shape, 1)))[0][target]
print(f"Prediction on target {target}: {pred_image}")
print(f"Original prediction on target {target}: {pred_image_og}")
print(f"Difference of {pred_image_og-pred_image}")