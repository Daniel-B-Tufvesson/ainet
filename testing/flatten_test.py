import numpy as np

import layers
import ainet


def test_1():

    from_shape = (125, 125)
    to_shape = (125*125, )

    model = ainet.Sequence(input_dimensions=from_shape)
    model.add(layers.Flatten())
    model.compile()

    x = np.arange(0, to_shape[0]).reshape(from_shape)
    y = model.predict(x)
    assert y.shape == to_shape

    print('test passed')


if __name__ == '__main__':
    test_1()

