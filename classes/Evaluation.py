import numpy as np
import PIL
from classes.DNN_Models import MnistClassifier, FlowerClassifier


class Evaluate:
    def evaluate(self):
        pass


class Ackley(Evaluate):
    def __init__(self, a = 20, b = 0.2, c = 2*np.pi):
        self.a = a
        self.b = b
        self.c = c
        self.optimum = 0
        
    def evaluate(self, x):
        dim = len(x)
        term1 = -1. * self.a * np.exp(-1. * self.b * np.sqrt((1./dim) * sum(map(lambda i: i**2, x))))
        term2 = -1. * np.exp((1./dim) * (sum(map(lambda j: np.cos(self.c * j), x))))
        
        return (term1 + term2 + self.a + np.exp(1))


class Rastringin:
    def __init__(self, a = 10):
        self.a = a
        self.optimum = 0
        
    def evaluate(self, x):
        y = self.a * len(x) + sum(map(lambda i: i**2 - self.a * np.cos(2*np.pi*i), x))
        return y


class ClassifierCrossentropy(Evaluate):

    models = { 'simple_classifier': FlowerClassifier(),
                'mnist_classifier': MnistClassifier()
            }


    def __init__(self, model, img_path, img_class, epsilon=0.05):
        
        self.model = self.models[model]
        
        # We open this image onece for performance, 
        # instead of doing it in the evaluation function
        img = PIL.Image.open(img_path)
        img = img.resize((self.model.img_width,self.model.img_height), resample=0)
        img_array = np.array(img)
        self.img = np.expand_dims(img_array,axis=0)
        self.img_class = img_class
        self.classes = self.model.class_names
        self.class_idx = self.classes.index(img_class)
        self.epsilon = epsilon


    def evaluate(self,x):
        processed_img = self.img + np.expand_dims(np.array(self.epsilon*x*255. ,dtype=np.uint8),axis=0).reshape(self.img.shape)
        processed_img = processed_img.clip(0, 255)
        preds = self.model.model.predict(processed_img)[0]
        eval = -np.log(preds[self.class_idx])
        return eval

    def predict(self,x):
        processed_img = self.img + np.expand_dims(np.array(self.epsilon*x*255. ,dtype=np.uint8),axis=0).reshape(self.img.shape)
        processed_img = processed_img.clip(0, 255)
        preds = self.model.model.predict(processed_img)
        return preds

    




