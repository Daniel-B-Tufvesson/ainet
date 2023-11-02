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

    def backward_pass(self, gradients: np.ndarray) -> np.ndarray:
        # Back-propagate through the layers.
        for layer in reversed(self.layers):
            gradients = layer.backward_pass(gradients)

        return gradients

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward_pass(x)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=1, learning_rate=0.01):
        #x = x.astype(dtype=float)
        y = y[:, np.newaxis]

        # Todo: gets stuck in local minimum. Fix: Momentum? Adam?


        for epoch in range(epochs):
            prediction = self.forward_pass(x)
            loss, error = self.compute_loss(prediction, y)
            loss_gradients = self.compute_loss_gradients(error)

            #print('loss', loss, 'error', error.shape, 'loss_gradients', loss_gradients.shape)

            print(f'epoch: {epoch+1}/{epochs}.  loss: {loss}')
            #print('loss gradients: ', loss_gradients.shape)

            # Backpropagate the loss gradients.
            self.backward_pass(loss_gradients)

            # Update the weights.
            self.update_parameters(learning_rate)



    def compute_loss(self, prediction_y: np.ndarray, target_y: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Compute the loss and error.
        :param prediction_y:
        :param target_y:
        :return:
        """
        #print('prediction: ', prediction_y.shape, ', target: ', target_y.shape)
        #error = (target_y - prediction_y.transpose()).transpose() # Kinda weird that we have to transpose.
        error = target_y - prediction_y
        #print('error: ', error.shape)
        loss = np.mean(error ** 2)
        return loss, error


    def compute_loss_gradients(self, error: np.ndarray) -> np.ndarray:
        n = error.shape[0]
        return 2 * error / n

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

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
    model.add(layers.Sigmoid())
    model.add(layers.FullyConnected(5))
    model.add(layers.Sigmoid())
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

    prediction = model.predict(train_data_x)
    #print(prediction)

    # Train the model.
    print('train model...')
    model.fit(train_data_x, train_data_y, epochs=100)
    print('training complete')



if __name__ == '__main__':
    test_train_seq_model()
