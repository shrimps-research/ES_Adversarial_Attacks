import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class SimpleClassifier:

    def __init__(self,model_path):

        self.model_path = model_path
        self.batch_size = 32
        self.img_height = 180
        self.img_width = 180
        class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        self.num_classes = len(class_names)

        self.data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                    input_shape=(
                                                    self.img_height,
                                                    self.img_width,
                                                    3
                                                )
                                ),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )


        self.model = tf.keras.models.load_model(self.model_path)