import gzip
import pickle

import numpy as np

def load_data(path="data/mnist.pkl.gz"):
    """Returns training, validation, and test data as they are in raw format."""
    with gzip.open(path, "rb") as file:
        training_data, validation_data, test_data = pickle.load(file, encoding="latin1")
    return training_data, validation_data, test_data


def load_training_batch(path="data/mnist.pkl.gz", batch_size=64):
    """Returns a sinlge training batch (using network's column format)."""
    training_data, _, _ = load_data(path)
    images, labels = training_data
    images = np.asarray(images, dtype=float)
    labels = np.asarray(labels)
    batch_images = images[:batch_size].T
    batch_labels = labels[:batch_size]
    return batch_images, batch_labels