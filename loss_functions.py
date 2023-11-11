from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):

    @abstractmethod
    def loss(self, prediction_y: np.ndarray, target_y: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, prediction_y: np.ndarray, target_y: np.ndarray) -> np.ndarray:
        pass


class MSE(LossFunction):
    def loss(self, prediction_y, target_y):
        return np.mean((prediction_y - target_y) ** 2)

    def gradient(self, prediction_y, target_y):
        if len(prediction_y.shape) != 2:
            raise ValueError(f'input must have a shape of length 2. got: {prediction_y.shape}')

        n = prediction_y.shape[0]
        return 2 * np.sum(prediction_y - target_y, axis=0) / n


class MAE(LossFunction):
    """
    The Mean Absolute Error loss function.
    """

    def loss(self, prediction_y: np.ndarray, target_y: np.ndarray) -> float:
        mae = np.mean(np.abs(prediction_y - target_y))
        return mae

    def gradient(self, prediction_y: np.ndarray, target_y: np.ndarray) -> np.ndarray:
        mae_gradient = np.where(prediction_y > target_y, 1, -1)
        return mae_gradient


class CategoricalCrossEntropy(LossFunction):
    # NOTE: Does not work. We get weird number overflows.

    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    @staticmethod
    def log_softmax(x: np.ndarray) -> np.ndarray:

        assert not np.isnan(x).any()
        assert not np.isinf(x).any()

        # Subtract the maximum value for numerical stability
        x = x - np.max(x, axis=-1, keepdims=True)

        assert not np.isnan(x).any()

        # Compute the log-softmax using the log-sum-exp trick
        log_softmax_probs = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

        assert not np.isnan(log_softmax_probs).any()

        return log_softmax_probs

    @staticmethod
    def softmax_cross_entropy_with_logits(target_y, logits):
        return -np.sum(target_y * (logits + np.log(np.sum(np.exp(logits),
                                                          axis=-1, keepdims=True))))

    def loss(self, prediction_y: np.ndarray, target_y: np.ndarray) -> float:
        # We assume the target_y are always given in probabilities.

        assert not np.isnan(prediction_y).any()

        # Compute cross entropy from logits using softmax.
        if self.from_logits:
            """
            log_softmax_probs = self.log_softmax(prediction_y)

            assert not np.isnan(log_softmax_probs).any()

            cross_entropy = -np.sum(target_y * log_softmax_probs) / len(log_softmax_probs)

            #assert not np.isnan(cross_entropy).any()

            return cross_entropy
            """
            return self.softmax_cross_entropy_with_logits(target_y, prediction_y)

        # Compute the cross entropy from ordinary probabilities.
        else:
            cross_entropy = -np.sum(target_y * np.log(prediction_y)) / len(prediction_y)

            assert not np.isnan(cross_entropy).any()

            return cross_entropy

    def gradient(self, prediction_y, target_y) -> np.ndarray:

        # Compute gradient from logits.
        if self.from_logits:
            log_softmax_probs = self.log_softmax(prediction_y)
            return log_softmax_probs - target_y

        # Compute gradient from probabilities.
        else:
            return prediction_y - target_y


class CategoricalCrossEntropyV2(LossFunction):

    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    @staticmethod
    def log_softmax(x: np.ndarray) -> np.ndarray:
        # Subtract the maximum value for numerical stability
        x = x - np.max(x, axis=-1, keepdims=True)

        # Compute the log-softmax using the log-sum-exp trick
        return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

    def loss(self, prediction_y: np.ndarray, target_y: np.ndarray) -> float:

        # Make sure the input has the right dimensions.
        if len(prediction_y.shape) != 2:
            raise ValueError(f'input must have a shape of length 2. got: {prediction_y.shape}')

        # Compute cross entropy from logits using softmax.
        if self.from_logits:
            log_softmax_probs = self.log_softmax(prediction_y)
            return -np.sum(target_y * log_softmax_probs) / len(log_softmax_probs)

        else:
            raise NotImplementedError()

    def gradient(self, prediction_y: np.ndarray, target_y: np.ndarray) -> np.ndarray:

        # Make sure the input has the right dimensions.
        if len(prediction_y.shape) != 2:
            raise ValueError(f'input must have a shape of length 2. got: {prediction_y.shape}')

        if self.from_logits:
            log_softmax_probs = self.log_softmax(prediction_y)
            gradient = np.sum(log_softmax_probs - target_y, axis=0) / len(prediction_y)
            return gradient

        else:
            raise NotImplementedError()