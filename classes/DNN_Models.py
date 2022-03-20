import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class FlowerClassifier:
    def __init__(self):
        self.batch_size = 32
        self.img_height = 128
        self.img_width = 128
        self.class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        self.num_classes = len(self.class_names)
        self.model = tf.keras.models.load_model('../data/dnn_models/flower_classifier')


class MnistClassifier:
    def __init__(self):
        self.batch_size = 32
        self.img_height = 28
        self.img_width = 28
        self.class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.num_classes = len(self.class_names)
        self.model = tf.keras.models.load_model('../data/dnn_models/mnist_classifier')
