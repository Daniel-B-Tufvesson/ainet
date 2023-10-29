from abc import ABC, abstractmethod

import numpy as np

import layers


class BaseModel(layers.Layer, ABC):

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray, y:np.ndarray):
        pass

    @abstractmethod
    def compile(self):
        """Compile the model."""
        pass


class Sequence(BaseModel):
    """
    A sequence of layers.
    """

    def __init__(self, input_dimensions: tuple[int, ...], layers=None):
        super().__init__()

        self.input_dimensions = input_dimensions

        if layers is None:
            self.layers = []  # type: list[layers.Layer]
        else:
            self.layers = layers

    def add(self, layer: layers.Layer):
        self.layers.append(layer)

    def compile(self):
        self.compile_layer(self.input_dimensions)

    def compile_layer(self, input_dimensions: tuple[int, ...]):
        super().compile_layer(input_dimensions)

        # Compile each layer.
        for layer in self.layers:
            layer.compile_layer(input_dimensions)
            input_dimensions = layer.output_dimensions

        # Last layers output dimensions becomes this sequence's output dimensions.
        self.output_dimensions = input_dimensions

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        transforms = x

        # Pass through all layers.
        for layer in self.layers:
            transforms = layer.forward_pass(transforms)

        return transforms

    def _compute_local_gradients(self, gradients: np.ndarray) -> np.ndarray:
        # Back-propagate through the layers.
        for layer in reversed(self.layers):
            gradients = layer.backward_pass(gradients)

        return gradients

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward_pass(x)

    def fit(self, x: np.ndarray, y: np.ndarray):
        print('fit not implemented')

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        print('evaluate not implemented.')


def test_sequence_model():
    model = Sequence(input_dimensions=(10, ))
    model.add(layers.FullyConnected(10))
    model.add(layers.ReLU())
    model.add(layers.FullyConnected(2))
    model.compile()

    x = np.array([0, 1, 2, 1, 0, -2, 2, 0, 1, -4])
    print(model.predict(x))


def test_train_seq_model():

    input_len = 5

    model = Sequence(input_dimensions=(input_len,))
    model.add(layers.FullyConnected(input_len))
    model.add(layers.ReLU())
    model.add(layers.FullyConnected(1))
    model.compile()

    # Generate training and validation data.
    import random as rand
    import math

    def new_sample_x():
        return [rand.randrange(1, 10) for _ in range(input_len)]

    train_data_x = np.array([new_sample_x() for _ in range(100)])
    train_data_y = np.array([math.prod(x) for x in train_data_x])

    val_data_x = np.array([new_sample_x() for _ in range(100)])
    val_data_y = np.array([math.prod(x) for x in val_data_x])

    # Train the model.
    print('train model...')
    model.fit(train_data_x, train_data_y)
    print('training complete')



if __name__ == '__main__':
    test_train_seq_model()
