import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class SimpleClassifier:

    def __init__(self,model_path, 
                img_height=128, 
                img_width=128,
                batch_size=32
                ):

        self.model_path = model_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
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
                layers.RandomFlip('horizontal'),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.1),
            ]
        )


        self.model = tf.keras.models.load_model(self.model_path)