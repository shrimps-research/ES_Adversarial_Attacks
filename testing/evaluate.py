import sys
from PIL import Image
import numpy as np

# setting paths to folders
sys.path.append('..')
from ES_adversarial_attacks import DNN_Models

# load original image
# original_img = Image.open("../data/img_data/perc_224/tench.png").convert("RGB")
# original_img = Image.open("../data/img_data/xcept_299/tench.png").convert("RGB")
original_img = Image.open("../results/noisy_input.png").convert("RGB")

# resize (vit test)
# from torchvision import transforms
# original_img = transforms.Compose([transforms.Resize((384,384))])(original_img)

original_img = np.array(original_img) / 255.0
if len(original_img.shape) == 2:
    original_img = np.expand_dims(original_img, axis=2)
# load model
model = DNN_Models.PerceiverClassifier()

# eval input
pred = model(original_img)[0].numpy()
pred_ind = np.argmax(pred)
print(f"Prediction: {pred[pred_ind]} on class {pred_ind}")
print(f"Prediction: {pred[0]} on class {0}")