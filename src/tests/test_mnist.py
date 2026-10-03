import gzip
import pickle

import numpy as np

from digits.mnist import load_data, load_training_batch


def create_dummy_mnist_file(path):
    training_data = (
        np.arange(12, dtype=float).reshape(3, 4),
        np.array([0, 1, 2]),
    )
    validation_data = (
        np.arange(8, dtype=float).reshape(2, 4),
        np.array([3, 4]),
    )
    test_data = (
        np.arange(4, dtype=float).reshape(1, 4),
        np.array([5]),
    )
    with gzip.open(path, "wb") as file:
        pickle.dump(
            (training_data, validation_data, test_data),
            file,
        )


def test_load_data(tmp_path):
    dataset_path = tmp_path / "mnist.pkl.gz"
    create_dummy_mnist_file(dataset_path)
    training_data, validation_data, test_data = load_data(dataset_path)
    images, labels = training_data
    assert images.shape == (3, 4)
    assert labels.tolist() == [0, 1, 2]
    assert validation_data[0].shape == (2, 4)
    assert test_data[0].shape == (1, 4)


def test_load_training_batch_returns_column_format(tmp_path):
    dataset_path = tmp_path / "mnist.pkl.gz"
    create_dummy_mnist_file(dataset_path)
    images, labels = load_training_batch(
        path=dataset_path,
        batch_size=2,
    )
    assert images.shape == (4, 2)
    assert labels.shape == (2,)
    expected_images = np.array([
        [0.0, 4.0],
        [1.0, 5.0],
        [2.0, 6.0],
        [3.0, 7.0],
    ])
    np.testing.assert_allclose(images, expected_images)
    np.testing.assert_array_equal(labels, np.array([0, 1]))