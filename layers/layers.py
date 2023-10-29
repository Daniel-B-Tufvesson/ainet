from abc import abstractmethod, ABC

import numpy as np
import random as rand


class Layer(ABC):

    def __init__(self):
        self.x = np.empty(0)  # The input to this layer.
        self.y = np.empty(0)  # The output of this layer.
        self.input_dimensions = (0,)  # The dimensionality of the input.
        self.output_dimensions = (0,)  # The dimensionality of the output.

    def compile_layer(self, input_dimensions: tuple[int, ...]):
        """
        Compile this layer. This is usually where the layer creates its parameters.
        :param input_dimensions: the dimensionality of the input.
        """
        self.input_dimensions = input_dimensions
        self.output_dimensions = input_dimensions

    def forward_pass(self, x: np.ndarray) -> np.ndarray:

        if self.input_dimensions != x.shape:
            raise ValueError(f'Input array does not have correct dimensions. Expected: '
                             f'{self.input_dimensions}, got: {x.shape}')

        y = self._compute_forward(x)
        self.x = x
        self.y = y
        return y

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        local_gradients = self._compute_local_gradients(gradients)
        return local_gradients * gradients

    @abstractmethod
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass from the input.
        :param x: the input.
        :return: the output of this layer.
        """
        pass

    @abstractmethod
    def _compute_local_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        Compute the local gradients with respect to the input.
        :param gradients: the upstream gradients.
        :return: the local gradients.
        """
        pass


class FullyConnected(Layer):
    """
    A fully connected neural layer, with weights connected from each input element to each
    output element. The input must be one-dimensional.
    NOTE: this is purely linear. No activation functions are used.
    """

    def __init__(self, units: int):
        super().__init__()
        self.units = units  # The number of units in this layer.
        self.weights = np.zeros(0)  # The weight matrix.

    def compile_layer(self, input_dimensions: tuple[int, ...]):

        # Make sure input vector only has one dimension.
        if len(input_dimensions) != 1:
            raise ValueError('fully connected layers only support 1 dimensional input')

        super().compile_layer(input_dimensions)

        # Create the weight matrix.
        self.weights = np.zeros(shape=(self.units, input_dimensions[0]))
        self.rand_init_weights()

        # The output is one dimensional with an element for each unit.
        self.output_dimensions = (self.units, )

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        print(x.shape)
        # Matrix multiplication.
        return self.weights @ x

    def _compute_local_gradients(self, gradients: np.ndarray) -> np.ndarray:
        pass

    def rand_init_weights(self):
        """
        Initialize weights with small random numbers. Suitable only for shallow networks.
        """
        # Init with values in range [-1.0, 1.0]

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] = rand.random() * 2 - 1


class ReLU(Layer):

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def _compute_local_gradients(self, gradients: np.ndarray) -> np.ndarray:
        return np.ndarray([1 if x_i > 0 else 0 for x_i in self.x])

