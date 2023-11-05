import gzip

import numpy as np

import activations
import ainet as ai
import layers
import loss_functions
import optimizers


def load_labels(filename: str, max_examples: int|None = None) -> np.ndarray:
    with gzip.open(filename) as labels:
        label_data = labels.read()

        examples_count = len(label_data)
        if max_examples is not None:
            examples_count = min(examples_count, max_examples + 8)

        # Note: the labels start at position 8.
        return np.array([label_data[i] for i in range(8, examples_count)])


def load_images(filename: str, max_examples: int|None = None) -> np.ndarray:
    with gzip.open(filename) as images:
        image_data = images.read()

        # Get the image count.
        image_count = image_data[4] << 24 | image_data[5] << 16 | image_data[6] << 8 | image_data[7]
        if max_examples is not None:
            image_count = min(image_count, max_examples)
        print('IMAGE_COUNT', image_count)

        rows = image_data[8] << 24 | image_data[9] << 16 | image_data[10] << 8 | image_data[11]
        print('ROWS: ', rows)
        columns = image_data[12] << 24 | image_data[13] << 16 | image_data[14] << 8 | image_data[15]
        print('COLUMNS: ', columns)

        def image_vector(start_pos) -> list:
            return [float(image_data[start_pos + i]) / 255.0 for i in range(rows * columns)]

        image_vectors = [image_vector(16 + i * rows * columns) for i in range(image_count)]
        return np.array(image_vectors)


# Load the data
TRAIN_EXAMPLE_COUNT = 100
train_x = load_images('data/train-images-idx3-ubyte.gz', max_examples=TRAIN_EXAMPLE_COUNT)
train_y = load_labels('data/train-labels-idx1-ubyte.gz', max_examples=TRAIN_EXAMPLE_COUNT)

print('Training examples: x:', train_x.shape[0], 'y:', train_y.shape[0])


# Create the model.

IMAGE_SIZE = 28
CLASS_COUNT = 10

model = ai.Sequence(input_dimensions=(IMAGE_SIZE**2, ))
model.add(layers.FullyConnected(20))
model.add(activations.ReLU())
model.add(layers.FullyConnected(20))
model.add(activations.ReLU())
model.add(layers.FullyConnected(CLASS_COUNT))

model.compile(optimizer=optimizers.SGDMomentum())


# Train the model.
model.fit(train_x, train_y, epochs=100, learning_rate=0.001,
          loss_func=loss_functions.CategoricalCrossEntropy(from_logits=True))

# Post-train test.
for x, y in zip(train_x[:10], train_y[:10]):
    print(model.predict(x), ', expected: ', y)

