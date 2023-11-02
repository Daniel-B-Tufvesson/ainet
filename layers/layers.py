from abc import abstractmethod, ABC

import numpy as np
import random as rand


class Layer(ABC):

    def __init__(self):
        self.x = np.empty(0)  # The last input to this layer.
        self.y = np.empty(0)  # The last output of this layer.
        #self.gradients = np.empty(0)  # The last (non-local) gradient of this layer.
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
        """
        Compute the forward pass.
        :param x: a matrix with shape (input_dimensions) for unbatched, or (n, input_dimensions) for batched with
        a batch size of n.
        :return:
        """

        # Check if this is a batched input.
        if len(x.shape) == len(self.input_dimensions) + 1:

            if self.input_dimensions != x.shape[1:]:
                raise ValueError(f'Input array does not have correct dimensions. Expected: '
                                 f'{self.input_dimensions}, got: {x.shape[0:]}')

            # Make the row-vectors into col-vectors.
            #x = x.transpose()

        # Check unbatched input.
        else:
            if self.input_dimensions != x.shape:
                raise ValueError(f'Input array does not have correct dimensions. Expected: '
                                 f'{self.input_dimensions}, got: {x.shape}')

            # Promote vector to matrix.
            #x = x[:, np.newaxis]


            #x = x.transpose()  # We want it to be a row vector.
            #print(x.shape, ", pre-transpose: ", x.transpose().shape)

        y = self._compute_forward(x)
        self.x = x
        self.y = y
        return y

    @abstractmethod
    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        pass
        #local_gradients = self._compute_local_gradients(gradients)
        #self.gradients = local_gradients * gradients  # Elementwise mult. correct approach?
        #return self.gradients

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
    output element. The input must be one-dimensional.
    NOTE: this is purely linear. No activation functions are used.
    """

    def __init__(self, units: int):
        super().__init__()
        self.units = units  # The number of units in this layer.
        self.weights = np.zeros(0)  # The weight matrix.
        self.weight_gradients = np.zeros(0)  # The gradients for each weight.

    def compile_layer(self, input_dimensions: tuple[int, ...]):

        # Make sure input vector only has one dimension.
        if len(input_dimensions) != 1:
            raise ValueError('fully connected layers only support 1 dimensional input')

        super().compile_layer(input_dimensions)

        # Create the weight matrix.
        self.weights = np.zeros(shape=(input_dimensions[0], self.units))
        self.rand_init_weights()

        # The output is one dimensional with an element for each unit.
        self.output_dimensions = (self.units, )

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        self.weight_gradients = self.x.transpose() @ gradients
        x_gradients = gradients @ self.weights.transpose()
        #print('x: ', self.x)
        #print('dy:', gradients)
        #print('dw', self.weight_gradients)

        return x_gradients

    def update_parameters(self, learning_rate):
        # SGD optimization.
        self.weights += self.weight_gradients * learning_rate  # Should this not be minus??
        #print('dW: ', self.weight_gradients)
        #print('W: ', self.weights)

    def rand_init_weights(self):
        """
        Initialize weights with small random numbers. Suitable only for shallow networks.
        """
        # Init with values in range [-1.0, 1.0]

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] = (rand.random() * 2 - 1) * 0.0001

        #print('rand weights: ', self.weights)




class ReLU(Layer):

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:

        #relu_gradient = np.vectorize(lambda x: 1 if x > 0 else 0)

        #return relu_gradient(gradients)
        return gradients * (self.x > 0)

        #return np.ndarray([1 if x_i > 0 else 0 for x_i in self.x])

    def update_parameters(self, learning_rate):
        pass


class Sigmoid(Layer):

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x)

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        s = self.sigmoid(self.x)
        da = s * (1 - s)
        return da * gradients

    def update_parameters(self, learning_rate):
        pass