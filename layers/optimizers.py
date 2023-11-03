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
