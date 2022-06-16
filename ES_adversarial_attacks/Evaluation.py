from scipy.stats import entropy
from PIL import Image
import numpy as np


class Evaluation:
    def worst_eval(self, minimize=True):
        """ Return worst possible evaluation
            That will be the initial value of the best evaluation in the ES
        """
        return np.inf if minimize else -np.inf

    def evaluate(self):
        pass


class Ackley(Evaluation):
    """ Evaluate a solution on Ackley problem
    """
    def __init__(self, a = 20, b = 0.2, c = 2*np.pi, minimize=True):
        self.a = a
        self.b = b
        self.c = c
        self.optimum = 0
        
    def evaluate(self, x):
        dim = len(x)
        term1 = -1. * self.a * np.exp(-1. * self.b * np.sqrt((1./dim) * sum(map(lambda i: i**2, x))))
        term2 = -1. * np.exp((1./dim) * (sum(map(lambda j: np.cos(self.c * j), x))))
        
        return (term1 + term2 + self.a + np.exp(1))


class Rastringin(Evaluation):
    """ Evaluate a solution on Rastringin problem
    """
    def __init__(self, a = 10, minimize=True):
        self.a = a
        self.optimum = 0
        
    def evaluate(self, x):
        y = self.a * len(x) + sum(map(lambda i: i**2 - self.a * np.cos(2*np.pi*i), x))
        return y


class Crossentropy(Evaluation):
    """ Classic crossentropy evaluation.
    """
    def __init__(self, model, true_label, minimize=True, targeted=False):
        self.model = model
        self.true_label = int(true_label)
        self.minimize = minimize
        self.targeted = targeted

    def worst_eval(self):
        """ Return worst evaluation possible for the current problem configutation.
        """
        if not self.targeted:
            return 0
        else:
            return np.inf if self.minimize else -np.inf

    def evaluate_ind(self, noise, input_):
        """ Deprecated.
            If targeted attack, use crossentropy (-log(pred)) on target.
            If untargeted attack, use negative crossentropy (log(pred)) on target.
            Opposite of above if minimize is False.
        """
        # feasible input space contraint
        # if np.min(input_) < 0 or np.max(input_) > 1:
        #     return np.inf if self.targeted else 0
        # prediction
        predictions = self.model(np.expand_dims(noise + input_, axis=0))[0].numpy()
        # return loss
        if self.minimize:
            loss_sign = (-1 if self.targeted else 1)
        else:
            loss_sign = (1 if self.targeted else -1)
        return loss_sign * np.log(predictions[self.true_label])

    def predict(self, batch):
        batch_size = min(32, batch.shape[0])
        return [self.model(b).numpy() for b in np.array_split(batch, batch.shape[0] / batch_size)]

    def evaluate(self, batch, pop_size, dataloader=False):
        """ If targeted attack, use crossentropy (-log(pred)) on target.
            If untargeted attack, use negative crossentropy (log(pred)) on target.
            Opposite of above if minimize is False.
            This evaluation function work on batches of individuals and images.
        """
        # feasible input space contraint
        # if np.min(input_) < 0 or np.max(input_) > 1:
        #     return np.inf if self.targeted else 0
        # compute model predictions
        if dataloader is None:  # batch contains all the noised images
            predictions = np.vstack(self.predict(batch))
        else:  # in this case batch contains only the noises
            predictions = []
            for og_batch in dataloader:
                og_batch = og_batch[0].numpy()
                noisy_batch = []
                for noise in batch:
                    noisy_batch.append((noise + og_batch).clip(0, 1))
                predictions += self.predict(np.vstack(noisy_batch))
            predictions = np.vstack(predictions)
        # compute loss for the entire batch
        if self.minimize:
            loss_sign = (-1 if self.targeted else 1)
        else:
            loss_sign = (1 if self.targeted else -1)
        # calculate accuracy
        pred_groups = predictions.argmax(axis=1).reshape((pop_size, int(predictions.shape[0]/pop_size)))
        # acc = (pred_groups==self.true_label).sum(axis=1)/pred_groups.shape[1]
        # get pred at target label
        predictions = predictions[:, self.true_label]
        # divide batch of predictions in groups associated to the individuals
        # and compute the mean prediction for each of these groups
        pred_groups = predictions.reshape((pop_size, int(predictions.shape[0]/pop_size)))
        predictions = pred_groups.mean(axis=1)
        # compute crossentropy loss for each ind mean prediction
        return loss_sign * np.log(predictions)  # - 0.1*entropy(pred_groups, axis=1) # * (1 + acc)


class BlindEvaluation(Evaluation):
    """ Evaluation in blind mode of a batch of input samples:
        the output of the model is only used to extract the batch accuracy information.
    """
    def __init__(self, model, true_label, minimize=True, targeted=False):
        self.model = model
        self.true_label = int(true_label)
        self.minimize = minimize
        self.targeted = targeted

    def worst_eval(self):
        if not self.targeted:
            return 1 if self.minimize else 0
        else:
            return 1 if self.minimize else 0

    def evaluate(self, batch, pop_size):
        """ If targeted attack, use crossentropy (-log(pred)) on target.
            If untargeted attack, use negative crossentropy (log(pred)) on target.
            Opposite of above if minimize is False.
        """
        # feasible input space contraint
        # if np.min(input_) < 0 or np.max(input_) > 1:
        #     return np.inf if self.targeted else 0
        # prediction
        batch_size = min(32, batch.shape[0])
        predictions = [self.model(b).numpy() for b in np.array_split(batch, batch.shape[0] / batch_size)]
        predictions = np.vstack(predictions)
        # compute loss for the entire batch
        if self.minimize:
            loss_sign = (-1 if self.targeted else 1)
        else:
            loss_sign = (1 if self.targeted else -1)
        # calculate accuracy
        pred_groups = predictions.argmax(axis=1).reshape((pop_size, int(predictions.shape[0]/pop_size)))
        acc = (pred_groups==self.true_label).sum(axis=1)/pred_groups.shape[1]
        
        if (self.targeted and self.minimize) or (not self.targeted and not self.minimize):
            acc = 1 - acc

        return loss_sign * acc