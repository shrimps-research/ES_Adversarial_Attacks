from PIL import Image
import numpy as np


class Evaluate:
    def worst_eval(self, minimize=True):
        """ Return worst possible evaluation
            That will be the initial value of the best evaluation in the ES
        """
        return np.inf if minimize else -np.inf

    def evaluate(self):
        pass


class Ackley(Evaluate):
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


class Rastringin:
    """ Evaluate a solution on Rastringin problem
    """
    def __init__(self, a = 10, minimize=True):
        self.a = a
        self.optimum = 0
        
    def evaluate(self, x):
        y = self.a * len(x) + sum(map(lambda i: i**2 - self.a * np.cos(2*np.pi*i), x))
        return y


class Crossentropy(Evaluate):
    """ Generic image classifier evaluator using cross entropy
    """
    def __init__(self, model, true_label, minimize=True, targeted=False):
        self.model = model
        self.true_label = int(true_label)
        self.minimize = minimize
        self.targeted = targeted

    def worst_eval(self):
        if not self.targeted:
            return 0
        else:
            return np.inf if self.minimize else -np.inf

    def evaluate_ind(self, noise, input_):
        """ if targeted attack, use crossentropy (-log(pred)) on target
            if untargeted attack, use negative crossentropy (log(pred)) on target
            opposite of above if minimize is False
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

    def evaluate(self, batch, pop_size):
        """ if targeted attack, use crossentropy (-log(pred)) on target
            if untargeted attack, use negative crossentropy (log(pred)) on target
            opposite of above if minimize is False
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
        # batch_loss = loss_sign * np.log(predictions[:, self.true_label])
        # get pred at target label
        predictions = predictions[:, self.true_label]
        # divide batch of predictions in groups associated to the individuals
        # and compute the mean prediction for each of these groups
        predictions = predictions.reshape((pop_size, int(predictions.shape[0]/pop_size))).mean(axis=1)
        # compute crossentropy loss for each ind mean prediction
        return loss_sign * np.log(predictions) * (1 + acc)


class BlindEvaluation(Evaluate):
    """ Generic image classifier evaluator using cross entropy
    """
    def __init__(self, model, true_label, minimize=True, targeted=False):
        self.model = model
        self.true_label = int(true_label)
        self.minimize = minimize
        self.targeted = targeted

    def worst_eval(self):
        if not self.targeted:
            return 0
        else:
            return np.inf if self.minimize else -np.inf

    def evaluate(self, batch, pop_size):
        """ if targeted attack, use crossentropy (-log(pred)) on target
            if untargeted attack, use negative crossentropy (log(pred)) on target
            opposite of above if minimize is False
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

        return loss_sign * acc

class CrossentropySimilarity(Evaluate):
    """ Generic image classifier evaluator using cross entropy
    """
    def __init__(self, model, true_label, minimize=True, targeted=False):
        self.model = model
        self.true_label = int(true_label)
        self.minimize = minimize
        self.targeted = targeted

    def worst_eval(self):
        if not self.targeted:
            return 0
        else:
            return np.inf if self.minimize else -np.inf

    def evaluate_ind(self, noise, input_):
        """ if targeted attack, use crossentropy (-log(pred)) on target
            if untargeted attack, use negative crossentropy (log(pred)) on target
            opposite of above if minimize is False
        """
        predictions = self.model(np.expand_dims(noise + input_, axis=0))[0].numpy()
        # return loss
        if self.minimize:
            loss_sign = (-1 if self.targeted else 1)
        else:
            loss_sign = (1 if self.targeted else -1)
        return loss_sign * (np.log(predictions[self.true_label]) + noise.sum())

    def evaluate(self, batch, individuals):
        """ if targeted attack, use crossentropy (-log(pred)) on target
            if untargeted attack, use negative crossentropy (log(pred)) on target
            opposite of above if minimize is False
        """
        # feasible input space contraint
        # if np.min(input_) < 0 or np.max(input_) > 1:
        #     return np.inf if self.targeted else 0
        # prediction
        predictions = self.model(batch).numpy()
        # return loss
        if self.minimize:
            loss_sign = (-1 if self.targeted else 1)
        else:
            loss_sign = (1 if self.targeted else -1)

        # calculate loss contributions
        pred_contrib = np.log(predictions[:, self.true_label])
        noise_contrib = np.log(np.sum(individuals, axis=1)/individuals.max(axis=1))
        
        return loss_sign * ( pred_contrib + 0.01*noise_contrib )