"""
Some activation layers.
"""

from abc import ABC

import numpy as np

import optimizers
from layers import Layer


class BaseActivation(Layer, ABC):
    """An abstract base activation layer."""

    def update_parameters(self, learning_rate):
        # Most activation functions do not have learnable parameters.
        pass


class ReLU(BaseActivation):

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (self.x > 0)


class LeakyReLU(BaseActivation):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        # max(x, x * alpha)
        return np.where(x > 0, x, self.alpha * x)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * np.where(self.x > 0, 1, self.alpha)


class PReLU(BaseActivation):

    def __init__(self, initial_alpha=0.01):
        super().__init__()
        self.initial_alpha = initial_alpha
        self.alpha = np.empty(0)  # One alpha for each incoming unit.
        self.alpha_gradients = np.empty(0)

    def compile_layer(self, input_dimensions: tuple[int, ...], optimizer: optimizers.Optimizer):
        super().compile_layer(input_dimensions, optimizer)

        self.alpha = np.full(input_dimensions, fill_value=self.initial_alpha)

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        self.alpha_gradients = np.sum((self.x <= 0) * gradients) / gradients.size
        return np.where(self.x > 0, gradients, self.alpha * gradients)

    def update_parameters(self, learning_rate):
        super().update_parameters(learning_rate)
        self.alpha = self.optimizer.step(self.alpha, self.alpha_gradients, learning_rate)


class ELU(BaseActivation):
    """
    The ELU activation function is a variant of the rectified linear unit (ReLU) that aims to
    address some of the limitations of ReLU, such as the dying ReLU problem. It introduces a
    small, non-zero gradient for negative inputs to keep neurons from becoming inactive
    during training.
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (np.where(self.x > 0, 1, self.alpha * np.exp(self.x)))


class Sigmoid(BaseActivation):

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        # s = self.sigmoid(self.x)
        d_sigmoid = self.y * (1 - self.y)
        return gradients * d_sigmoid


class ZeroSigmoid(BaseActivation):
    """
    The Zero-Centered Sigmoid (z-sigmoid). This is a variation of the sigmoid function
    that aims to address the issue of non-zero-centered outputs. In the z-sigmoid, the
    output is scaled and shifted, resulting in values ranging from -1 to 1, and it is
    centered around zero
    """

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return 2 * (1 / (1 + np.exp(-x))) - 1

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (0.5 * (1 + self.y) * (1 - self.y))


class Tanh(BaseActivation):

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (1 - np.tanh(self.y) ** 2)

