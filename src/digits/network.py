import numpy as np

def sigmoid(z):
    """A sigmoid activation function applied for each value in NumPy array (z) independently"""
    return 1.0 / (1.0 + np.exp(-z))

class Network:
    """A small feed-forward network."""

    def __init__(self, layer_sizes, seed):
        self.layer_sizes = layer_sizes

        np.random.seed(seed)
        self.weights = [
            np.random.normal(size=(next_size, curr_size))
            for curr_size, next_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        self.biases = [np.random.randn(size, 1) for size in layer_sizes[1:]]

    def forward(self, x):
        """Returns output of the network for one input vector (x)."""
        activation = x

        for weights, biases in zip(self.weights, self.biases):
            activation = sigmoid(weights @ activation + biases)

        return activation
