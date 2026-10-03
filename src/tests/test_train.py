import numpy as np
import pytest

import gzip
import pickle
from digits.train import (
    accuracy,
    avg_loss,
    one_hot,
    prepare_data,
)


def create_dummy_mnist_file(path):
    training_data = (
        np.arange(2 * 784, dtype=float).reshape(2, 784),
        np.array([2, 7])
    )
    validation_data = (
        np.arange(784, dtype=float).reshape(1, 784),
        np.array([4])
    )

    test_data = (
        np.arange(784, dtype=float).reshape(1, 784),
        np.array([9])
    )

    with gzip.open(path, "wb") as file:
        pickle.dump(
            (training_data, validation_data, test_data),file
        )
def test_one_hot():
    result = one_hot(3)
    expected = np.zeros((10, 1))
    expected[3, 0] = 1.0
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (10, 1)
    assert np.sum(result) == 1.0

def test_prepare_data_training(tmp_path):
    dataset_path = tmp_path / "mnist.pkl.gz"
    create_dummy_mnist_file(dataset_path)
    result = prepare_data(dataset_path, dataset="training")
    assert len(result) == 2
    image, target = result[0]
    assert image.shape == (784, 1)
    assert target.shape == (10, 1)
    np.testing.assert_allclose(
        image[:4],
        np.array([[0.0], [1.0], [2.0], [3.0]])
    )
    assert np.argmax(target) == 2


def test_prepare_data_validation(tmp_path):
    dataset_path = tmp_path / "mnist.pkl.gz"
    create_dummy_mnist_file(dataset_path)
    result = prepare_data(dataset_path, dataset="validation")
    assert len(result) == 1
    assert result[0][0].shape == (784, 1)
    assert np.argmax(result[0][1]) == 4


def test_prepare_data_test_data(tmp_path):
    dataset_path = tmp_path / "mnist.pkl.gz"
    create_dummy_mnist_file(dataset_path)
    result = prepare_data(dataset_path, dataset="test")
    assert len(result) == 1
    assert np.argmax(result[0][1]) == 9


def test_prepare_data_rejects_unknown_dataset(tmp_path):
    dataset_path = tmp_path / "mnist.pkl.gz"
    create_dummy_mnist_file(dataset_path)
    with pytest.raises(ValueError, match="Unknown dataset"):
        prepare_data(dataset_path, dataset="unknown")


class DummyNetwork:
    def __init__(self, predictions):
        self.predictions = predictions
        self.id = 0

    def forward(self, image):
        prediction = self.predictions[self.id]
        self.id += 1
        return prediction


def test_accuracy():
    data = [
        (np.zeros((4, 1)), one_hot(2)),
        (np.zeros((4, 1)), one_hot(7)),
        (np.zeros((4, 1)), one_hot(4)),
    ]

    network = DummyNetwork(
        [
            np.array([[0.0], [0.0], [5.0]]),
            np.array([[0.0], [3.0], [0.0]]),
            np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
        ]
    )
    result = accuracy(network, data)
    assert result == pytest.approx(2 / 3)


def test_avg_loss():
    data = [
        (
            np.zeros((784, 1)),
            one_hot(2),
        ),
    ]
    class FixedNetwork:
        def forward(self, image):
            return np.array([[1.0],
                [2.0],
                [3.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0]
            ])
    result = avg_loss(FixedNetwork(), data)
    assert np.isfinite(result)
    assert result > 0.0