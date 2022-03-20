import tensorflow as tf


class FlowerClassifier:
    def __init__(self):
        self.batch_size = 32
        self.img_height = 128
        self.img_width = 128
        self.class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        self.num_classes = len(self.class_names)
        self.model = tf.keras.models.load_model('../data/dnn_models/flower_classifier')

    def __call__(self, x):
        self.model(x)[0]


class MnistClassifier:
    def __init__(self):
        self.batch_size = 32
        self.img_height = 28
        self.img_width = 28
        self.class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.num_classes = len(self.class_names)
        self.model = self.build()
        self.model.load_weights("../data/imgclf_nn_ckpt/weights")

    def build(self):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation = 'softmax')
        ])
    
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
        model.compile(optimizer = 'adam', 
                    loss = loss_fn, 
                    metrics = ['accuracy'])
        return model

    def __call__(self, x):
        self.model(x)[0]

class XceptionClassifier:
    def __init__(self):
        self.model = tf.keras.applications.Xception(weights='imagenet', include_top=True, input_shape=(299,299,3))

    def __call__(self, x):
        self.model(x)[0].numpy()