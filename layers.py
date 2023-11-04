from abc import abstractmethod, ABC

import numpy as np
import random as rand
import optimizers


class Layer(ABC):
    """
    Base class for a neural network layer. A layer performs two types of computations: a forward pass and a
    backward pass. The forward pass takes a matrix as input and produces a new matrix as output. This is
    the primary computational task for the layer.

    The backward pass is used when training the layer and the rest of the network. As input, it takes a gradient
    of the loss function with respect to the layer's output from the forward pass. From this it computes the gradient
    w.r.t. the layer's input from the forward pass, and gives this as its output. If the layer has trainable parameters,
    then the gradients w.r.t. these are also computed and then stored for later updating the parameters, after the
    full backward pass is over.

    Note that a layer does not necessarily need to contain neural units.
    """

    def __init__(self):
        self.x = np.empty(0)  # The last input to this layer.
        self.y = np.empty(0)  # The last output of this layer.
        self.input_dimensions = (0,)  # The dimensionality of the input.
        self.output_dimensions = (0,)  # The dimensionality of the output.
        self.optimizer = None  # type: optimizers.Optimizer | None

    def compile_layer(self, input_dimensions: tuple[int, ...], optimizer: optimizers.Optimizer):
        """
        Compile this layer. This is usually where the layer creates its parameters.
        :param optimizer:
        :param input_dimensions: the dimensionality of the input.
        """
        self.input_dimensions = input_dimensions
        self.output_dimensions = input_dimensions
        self.optimizer = optimizer

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass. This stores the input and the computed output in the layer.
        :param x: a matrix with shape (input_dimensions) for unbatched, or (n, input_dimensions)
                  for batched with a batch size of n.
        :return: a batched or unbatched matrix of the computed output of this layer.
        """

        # Check if this is a batched input, and if it has the correct dimensions.
        if len(x.shape) == len(self.input_dimensions) + 1:

            if self.input_dimensions != x.shape[1:]:
                raise ValueError(f'Input array does not have correct dimensions. Expected: '
                                 f'{self.input_dimensions}, got: {x.shape[0:]}')

        # Check if unbatched input has correct dimensions.
        else:
            if self.input_dimensions != x.shape:
                raise ValueError(f'Input array does not have correct dimensions. Expected: '
                                 f'{self.input_dimensions}, got: {x.shape}')

        # Do forward pass computations.
        y = self._compute_forward(x)
        self.x = x
        self.y = y
        return y

    @abstractmethod
    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass from the input.
        :param x: the input.
        :return: the output of this layer.
        """
        pass

    @abstractmethod
    def update_parameters(self, learning_rate):
        """Update the parameters of this layer using its gradients.
        :param learning_rate:
        """
        pass


class FullyConnected(Layer):
    """
    A fully connected neural layer, with weights connected from each input element to each
    output element.
    NOTE: this is a purely linear. No activation functions are used.
    """

    def __init__(self, units: int):
        super().__init__()
        self.units = units  # The number of units in this layer.
        self.weights = np.zeros(0)  # The weight matrix.
        self.weight_gradients = np.zeros(0)  # The gradients for each weight.

    def compile_layer(self, input_dimensions: tuple[int, ...], optimizer: optimizers.Optimizer):

        # Make sure input vector only has one dimension.
        if len(input_dimensions) != 1:
            raise ValueError('fully connected layers only support 1 dimensional input')

        super().compile_layer(input_dimensions, optimizer)

        # Create the weight matrix.
        self.weights = np.zeros(shape=(input_dimensions[0], self.units))
        self.rand_init_weights()

        # The output is one dimensional with an element for each unit.
        self.output_dimensions = (self.units,)

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        self.weight_gradients = self.x.transpose() @ gradients
        return gradients @ self.weights.transpose()

    def update_parameters(self, learning_rate):
        if self.optimizer is None:
            raise ValueError('optimizer is not set')

        self.weights = self.optimizer.step(self.weights, self.weight_gradients, learning_rate)

    def rand_init_weights(self):
        """
        Initialize weights with small random numbers. Suitable only for shallow networks.
        """

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] = (rand.random() * 2 - 1) * 0.01


class ReLU(Layer):

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (self.x > 0)

    def update_parameters(self, learning_rate):
        pass


class LeakyReLU(Layer):

    def __init__(self):
        super().__init__()
        self.alpha = 0.01

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        # max(x, x * alpha)
        return np.where(x > 0, x, self.alpha * x)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * np.where(self.x > 0, 1, self.alpha)

    def update_parameters(self, learning_rate):
        pass


class ELU(Layer):
    """
    The ELU activation function is a variant of the rectified linear unit (ReLU) that aims to
    address some of the limitations of ReLU, such as the dying ReLU problem. It introduces a
    small, non-zero gradient for negative inputs to keep neurons from becoming inactive
    during training.
    """

    def __init__(self):
        super().__init__()
        self.alpha = 0.01

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (np.where(self.x > 0, 1, self.alpha * np.exp(self.x)))

    def update_parameters(self, learning_rate):
        pass


class Sigmoid(Layer):

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        # s = self.sigmoid(self.x)
        d_sigmoid = self.y * (1 - self.y)
        return gradients * d_sigmoid

    def update_parameters(self, learning_rate):
        pass


class ZeroSigmoid(Layer):
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

    def update_parameters(self, learning_rate):
        pass


class Tanh(Layer):

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        return gradients * (1 - np.tanh(self.y) ** 2)

    def update_parameters(self, learning_rate):
        pass
