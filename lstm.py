import numpy as np
import math

class LSTMCell:

    def __init__(self, size:int):
        self.size = size
        self.i = 0
        self.f = 0
        self.o = 0
        self.g = 0
        self.weights = np.zeros(size, 2) # 1st col for x_t, 2nd col for h_t1.

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (3, self.size, 1):
            raise ValueError(f"incorrect shape: {x.shape}")

        c_t1 = x[0, :]  # The cell state vector.
        hx = x[1:2, :]  # The stacked vectors h_{t-1} and x_t.

        sums = self.weights @ hx
        self.i = self.sigmoid(sums[0, :])
        self.f = self.sigmoid(sums[1, :])
        self.o = self.sigmoid(sums[2, :])
        self.g = self.tanh(sums[3, :])

        fc = self.f * c_t1
        ig = self.i * self.g
        c_t = fc + ig
        h_t = self.o * self.tanh(c_t)

        return np.stack((c_t, h_t))


    def sigmoid(self, x : np.ndarray) -> np.ndarray:
        pass

    def tanh(self, x : np.ndarray) -> np.ndarray:
        pass



