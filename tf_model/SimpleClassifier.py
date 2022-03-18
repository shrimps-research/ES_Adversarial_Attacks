import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class SimpleClassifier:

    def __init__(self):
        
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


        self.model = Sequential([
                            self.data_augmentation,
                            layers.Rescaling(1./255),
                            layers.Conv2D(16, 3, padding='same', activation='relu'),
                            layers.MaxPooling2D(),
                            layers.Conv2D(32, 3, padding='same', activation='relu'),
                            layers.MaxPooling2D(),
                            layers.Conv2D(64, 3, padding='same', activation='relu'),
                            layers.MaxPooling2D(),
                            layers.Dropout(0.2),
                            layers.Flatten(),
                            layers.Dense(128, activation='relu'),
                            layers.Dense(self.num_classes)
                            ])
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.model.load_weights('tf_model/model_weights/checkpoint.ckpt')