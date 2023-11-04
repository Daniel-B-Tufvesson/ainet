from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):

    @abstractmethod
    def step(self, parameters: np.ndarray, gradients: np.ndarray,
             learning_rate: float) -> np.ndarray:
        """Compute the new parameters from the current parameters and their gradients."""
        pass


class SGD(Optimizer):
    def step(self, parameters: np.ndarray, gradients: np.ndarray,
             learning_rate: float) -> np.ndarray:
        return parameters - gradients * learning_rate


class SGDMomentum(Optimizer):

    def __init__(self, momentum=0.9):
        self.velocity = np.empty(0)
        self.momentum = momentum

    def step(self, parameters: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:

        # Shape mismatch indicates first step, so initialize velocity at 0.
        if self.velocity.shape != parameters.shape:
            self.velocity = np.zeros_like(parameters)

        self.velocity = self.momentum * self.velocity - learning_rate * gradients

        return parameters + self.velocity
