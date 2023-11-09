import numpy as np

import metrics
import random


def test_accuracy():
    metric = metrics.Accuracy()

    class_count = 10

    def rand_pred():
        correct = random.randint(0, class_count - 1)
        return [1 if i == correct else 0 for i in range(class_count)]

    target = np.array([rand_pred() for _ in range(1000)])
    predictions = target * 0.5
    predictions[0] = rand_pred()

    result = metric.metric(prediction_y=target, target_y=predictions)
    print(result)


if __name__ == '__main__':
    test_accuracy()
