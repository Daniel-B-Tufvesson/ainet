from abc import ABC, abstractmethod

import numpy as np

import ainet


class LossFunction(ABC):

    @abstractmethod
    def loss(self, prediction_y, target_y) -> float:
        pass

    @abstractmethod
    def gradient(self, prediction_y, target_y) -> np.ndarray:
        pass


class MSE(LossFunction):
    def loss(self, prediction_y, target_y):
        return np.mean((prediction_y - target_y) ** 2)

    def gradient(self, prediction_y, target_y):
        n = prediction_y.shape[0]
        return 2 * (prediction_y - target_y) / n


class CategoricalCrossEntropy(LossFunction):

    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def loss(self, prediction_y, target_y) -> float:
        # We assume the target_y are always given in probabilities.

        # Convert prediction from logits to probabilities.
        if self.from_logits:
            prediction_y = ainet.softmax(prediction_y)

        # Compute the cross entropy.
        return -np.sum(target_y * np.log(prediction_y)) / len(prediction_y)

    def gradient(self, prediction_y, target_y) -> np.ndarray:

        # Convert prediction from logits to probabilities.
        if self.from_logits:
            prediction_y = ainet.softmax(prediction_y)

        return prediction_y - target_y



