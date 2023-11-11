import gzip

import numpy as np

import activations
import ainet as ai
import layers
import loss_functions
import optimizers
import metrics

IMAGE_SIZE = 28
CLASS_COUNT = 10


def load_labels(filename: str, max_examples: int | None = None) -> np.ndarray:
    with gzip.open(filename) as labels:
        label_data = labels.read()

        examples_count = len(label_data)
        if max_examples is not None:
            examples_count = min(examples_count, max_examples + 8)

        # Vectorize the labels.
        def to_vector(class_index):
            return [1 if i == class_index else 0 for i in range(CLASS_COUNT)]

        # Note: the labels start at position 8.
        return np.array([to_vector(label_data[i]) for i in range(8, examples_count)])
        #return np.array([label_data[i] for i in range(8, examples_count)])


def load_images(filename: str, max_examples: int | None = None) -> np.ndarray:
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
TRAIN_EXAMPLE_COUNT = 1000  # 100
train_x = load_images('data/train-images-idx3-ubyte.gz', max_examples=TRAIN_EXAMPLE_COUNT)
train_y = load_labels('data/train-labels-idx1-ubyte.gz', max_examples=TRAIN_EXAMPLE_COUNT)

assert not np.isnan(train_x).any()
assert not np.isnan(train_y).any()

print('Training examples: x:', train_x.shape, 'y:', train_y.shape)

# Zero center the data.
train_x = train_x * 2 - 1

print('x: ', train_x[0][:5])
print('y: ', train_y[0])

# Create the model.

model = ai.Sequence(input_dimensions=(IMAGE_SIZE ** 2,))
model.add(layers.FullyConnected(500))
model.add(activations.ReLU())
model.add(layers.FullyConnected(200))
model.add(activations.ReLU())
model.add(layers.FullyConnected(CLASS_COUNT))

model.compile(optimizer=optimizers.SGDMomentum())

# Train the model.
# loss_func = loss_functions.CategoricalCrossEntropyV2(from_logits=True)
# loss_func = loss_functions.MAE()
loss_func = loss_functions.MSE()
model.fit(train_x, train_y, epochs=100, learning_rate=0.001,
          loss_func=loss_func,
          eval_metrics=[metrics.Accuracy()]
          )

# Post-train test.
for x, y in zip(train_x[:10], train_y[:10]):
    print(np.argmax(model.predict(x)), ', expected: ', np.argmax(y))
