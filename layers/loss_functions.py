from abc import ABC, abstractmethod

import numpy as np


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