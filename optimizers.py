from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    An abstract base optimizer. An optimizer is used for optimizing the parameters of a layer. Each layer should have
    a unique instance of an optimizer, that is, an optimizer should not be shared between several layers.
    """

    def initialize(self, parameter_shape: tuple[int, ...]):
        pass


    @abstractmethod
    def step(self, parameters: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        """Compute the new parameters from the current parameters and their gradients."""
        pass


class SGD(Optimizer):
    def step(self, parameters: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        return parameters - gradients * learning_rate


class SGDMomentum(Optimizer):

    def __init__(self, momentum=0.9):
        self.velocity = np.empty(0)
        self.momentum = momentum

    def initialize(self, parameter_shape: tuple[int, ...]):
        super().initialize(parameter_shape)
        self.velocity = np.zeros(shape=parameter_shape)

    def step(self, parameters: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        self.velocity = self.momentum * self.velocity - learning_rate * gradients
        return parameters + self.velocity


class Adam(Optimizer):

    def __init__(self, beta_1=0.9, beta_2=0.999):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.moment_1 = np.empty(0)
        self.moment_2 = np.empty(0)
        self.step_count = 0

    def initialize(self, parameter_shape: tuple[int, ...]):
        super().initialize(parameter_shape)
        self.moment_1 = np.zeros(shape=parameter_shape)
        self.moment_2 = np.zeros(shape=parameter_shape)
        self.step_count = 0

    def step(self, parameters: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
        # Increment time step.
        self.step_count += 1

        # Update moving averages.
        self.moment_1 = self.beta_1 * self.moment_1 + (1 - self.beta_1) * gradients
        self.moment_2 = self.beta_2 * self.moment_2 + (1 - self.beta_2) * gradients * gradients

        # Bias correct the averages.
        unbias_1 = self.moment_1 / (1 - self.beta_1 ** self.step_count)
        unbias_2 = self.moment_2 / (1 - self.beta_2 ** self.step_count)

        # Update parameters.
        return parameters - learning_rate * unbias_1 / (np.sqrt(unbias_2) + 1e-7)



