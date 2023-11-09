from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """
    Base class for computing a metric of the network's performance.
    """

    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def metric(self, prediction_y: np.ndarray, target_y: np.ndarray) -> float:
        pass


class Accuracy(Metric):

    def __init__(self, name: str = 'accuracy'):
        super().__init__(name)

    def metric(self, prediction_y: np.ndarray, target_y: np.ndarray) -> float:
        n_correct = 0
        for predicted, target in zip(prediction_y, target_y):
            pred_index = predicted.argmax()
            target_index = target.argmax()

            if pred_index == target_index:
                n_correct += 1

        return n_correct / len(prediction_y)