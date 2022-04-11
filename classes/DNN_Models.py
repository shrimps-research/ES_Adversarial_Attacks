import tensorflow as tf
import numpy as np
import torch


class FlowerClassifier:
    def __init__(self):
        self.batch_size = 32
        self.img_height = 128
        self.img_width = 128
        self.class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        self.num_classes = len(self.class_names)
        self.model = tf.keras.models.load_model('../data/dnn_models/flower_classifier')

    def __call__(self, x):
        return self.model(x)[0]


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
        if len(x.shape) == 2:
            # add batch dim
            x = np.expand_dims(x, axis=0)
        return self.model(x)[0]


class XceptionClassifier:
    def __init__(self):
        self.model = tf.keras.applications.Xception(weights='imagenet', include_top=True, input_shape=(299,299,3))

    def __call__(self, x):
        if len(x.shape) == 3:
            # add batch dim
            x = np.expand_dims(x, axis=0)
        return self.model(x)[0]


class ViTClassifier:
    def __init__(self):
        from pytorch_pretrained_vit import ViT
        self.model = ViT('B_32_imagenet1k', pretrained=True).double()
        self.model.eval()

    def __call__(self, x):
        if len(x.shape) == 3:
            # transpose dims from HxWxC to CxHxW
            x = np.transpose(x, (2, 0, 1))
            # add batch dim
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 4:
            # transpose dims from BxHxWxC to BxCxHxW
            x = np.transpose(x, (0, 3, 1, 2))

        with torch.no_grad():
            return self.model(torch.tensor(x, dtype=torch.float64))[0]


class PerceiverClassifier:
    def __init__(self):
        from transformers import PerceiverFeatureExtractor, PerceiverForImageClassificationLearned
        self.feature_extractor = PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-learned")
        self.model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")
        self.model.eval()
    
    def __call__(self, x):
        if len(x.shape) == 3:
            # transpose dims from HxWxC to CxHxW
            x = np.transpose(x, (2, 0, 1))
            # add batch dim
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 4:
            # transpose dims from BxHxWxC to BxCxHxW
            x = np.transpose(x, (0, 3, 1, 2))

        # prepare input
        encoding = self.feature_extractor(list(x), return_tensors="pt")
        inputs = encoding.pixel_values
        # forward pass
        with torch.no_grad():
            outputs = self.model(inputs)
            return torch.nn.Softmax(dim=1)(outputs.logits).detach()[0]