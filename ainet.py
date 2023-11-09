from abc import ABC, abstractmethod

import numpy as np
import random

import activations
import loss_functions
import metrics
import optimizers
import layers
import copy


class BaseModel(layers.Layer, ABC):

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def compile(self, optimizer):
        """Compile the model.
        :param optimizer:
        """
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

    def compile(self, optimizer: optimizers.Optimizer = None):

        # Set default optimizer if not specified.
        if optimizer is None:
            optimizer = optimizers.SGDMomentum()

        self.compile_layer(self.input_dimensions, optimizer)

    def compile_layer(self, input_dimensions: tuple[int, ...],
                      optimizer: optimizers.Optimizer):
        super().compile_layer(input_dimensions, optimizer)

        # Compile each layer.
        for layer in self.layers:
            layer.compile_layer(input_dimensions, copy.deepcopy(optimizer))
            input_dimensions = layer.output_dimensions

        # Last layers output dimensions becomes this sequence's output dimensions.
        self.output_dimensions = input_dimensions

    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        transforms = x

        # Pass through all layers.
        for layer in self.layers:
            transforms = layer.forward_pass(transforms)

        return transforms

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        # Back-propagate through the layers.
        for layer in reversed(self.layers):
            gradients = layer.backward_pass(gradients)

        return gradients

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward_pass(x)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=1, learning_rate=0.001, batch_size=32,
            loss_func: loss_functions.LossFunction | None = None,
            val_x: np.ndarray = None, val_y: np.ndarray = None,
            eval_metrics: list[metrics.Metric] = None,
            shuffle_batches=False):

        if len(x.shape) != 2:
            raise ValueError('x is expected to have a two dimensional shape: (number_of_examples, input_length)')

        if (x.shape[1],) != self.input_dimensions:
            raise ValueError(f'x does not have the correct input length. expected:'
                             f' {self.input_dimensions}, got: {(x.shape[1],)}')

        # Todo: check val data.

        if eval_metrics is None:
            eval_metrics = []

        if loss_func is None:
            print('Loss function not specified. Use loss_functions.MSE')
            loss_func = loss_functions.MSE()

        # Make sure if 1-dim vector, that it is a column vector.
        if len(x.shape) == 1:
            y = y[:, np.newaxis]

        # Batch the input data.
        batches = self.create_batches(x, y, batch_size)

        for epoch in range(epochs):

            total_loss = 0

            # Accumulated training metrics.
            accumulated_metrics = {metric.name: 0 for metric in eval_metrics}

            # Shuffle the batches.
            if shuffle_batches:
                random.shuffle(batches)

            # Train on each batch.
            for batch in batches:
                bx = batch[0]
                by = batch[1]

                # Make the forward pass.
                prediction = self.forward_pass(bx)

                # Compute the loss and its gradients.
                loss = loss_func.loss(prediction, by)
                loss_gradients = loss_func.gradient(prediction, by)
                loss_gradients = np.tile(loss_gradients, (by.shape[0], 1))  # Assign loss gradient to each example.
                total_loss += loss

                # Backpropagate the loss gradients.
                self.backward_pass(loss_gradients)

                # Update the weights.
                self.update_parameters(learning_rate)

                # Accumulate the training metrics.
                for metric in eval_metrics:
                    accumulated_metrics[metric.name] += metric.metric(prediction, by)

            # Do a per-epoch evaluation.
            avg_loss = total_loss / len(batches)
            print(f'epoch: {epoch + 1}/{epochs}.  loss: {avg_loss}', end='')

            # Print metrics on the training data.
            for name, value in accumulated_metrics.items():
                print(f', {name}: {value / len(batches)}', end='')

            print()

    @staticmethod
    def create_batches(x: np.ndarray, y: np.ndarray, batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
        example_count = x.shape[0]
        batches = []
        for start in range(0, example_count, batch_size):
            batch_x = x[start: start + batch_size]
            batch_y = y[start: start + batch_size]
            batches.append((batch_x, batch_y))

        return batches

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        print('evaluate not implemented.')


def test_sequence_model():
    model = Sequence(input_dimensions=(10,))
    model.add(layers.FullyConnected(10))
    model.add(activations.ReLU())
    model.add(layers.FullyConnected(2))
    model.compile(None)

    x = np.array([0, 1, 2, 1, 0, -2, 2, 0, 1, -4])
    print(model.predict(x))


def test_train_seq_model():
    input_len = 2

    model = Sequence(input_dimensions=(input_len,))
    model.add(layers.FullyConnected(input_len))
    model.add(activations.Tanh())
    model.add(layers.FullyConnected(2))
    model.add(activations.Tanh())
    model.add(layers.FullyConnected(1))
    model.compile(optimizer=optimizers.SGDMomentum())

    # Teach the model the OR function.
    train_data_x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    train_data_y = np.array([
        0,
        1,
        1,
        0
    ])

    # Pre-train test.
    for x, y in zip(train_data_x, train_data_y):
        print(model.predict(x), ', expected: ', y)

    model.fit(train_data_x, train_data_y, epochs=50, learning_rate=0.001)

    # Post-train test.
    for x, y in zip(train_data_x, train_data_y):
        print(model.predict(x), ', expected: ', y)

    """
    # Generate training and validation data.
    import random as rand
    import math

    def new_sample_x():
        return [rand.randrange(-10, 10) for _ in range(input_len)]

    train_data_x = np.array([new_sample_x() for _ in range(100)])
    train_data_y = np.array([math.prod(x) for x in train_data_x])

    val_data_x = np.array([new_sample_x() for _ in range(100)])
    val_data_y = np.array([math.prod(x) for x in val_data_x])

    prediction = model.predict(train_data_x)
    # print(prediction)

    # Train the model.
    print('train model...')
    model.fit(train_data_x, train_data_y, epochs=500)
    print('training complete')

    # print(model.create_batches(train_data_x, 32))

    print('Testing...')
    for i in range(20):
        print(model.predict(val_data_x[i]), ', expected: ', val_data_y[i])
    
    """


if __name__ == '__main__':
    test_train_seq_model()


def softmax(x: np.ndarray) -> np.ndarray:
    # Subtract the maximum value for numerical stability
    x = x - np.max(x, axis=-1, keepdims=True)

    # Calculate the exponential of each element in the input array
    exp_x = np.exp(x)

    # Sum the exponentials for normalization.
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)

    # Compute the softmax probabilities by dividing each exponential by the sum
    return exp_x / sum_exp_x


def log_softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=-1, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))


