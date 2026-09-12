import numpy as np

def sigmoid(z):
    """A sigmoid activation function (numerically stable version) applied for each value in NumPy array (z) independently"""
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    pos_values = z >= 0
    neg_values = ~pos_values

    result[pos_values] = 1.0 / (1.0 + np.exp(-z[pos_values]))
    exp_z = np.exp(z[neg_values])
    result[neg_values] = exp_z / (1.0 + exp_z)

    return result

def softmax(logits):
    """Softmax function (numerically stable version) that converts logits into probabilities"""
    shift_logits = logits - np.max(logits, axis=0, keepdims=True)
    exponentials = np.exp(shift_logits)
    return exponentials / np.sum(exponentials, axis=0, keepdims=True)

class Network:
    """A small feed-forward network."""

    def __init__(self, layer_sizes, seed):
        self.layer_sizes = layer_sizes

        self.rng = np.random.default_rng(seed)

        self.weights = [
            self.rng.normal(scale=1.0 / np.sqrt(curr_size), size=(next_size, curr_size))
            for curr_size, next_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        self.biases = [np.zeros((size, 1)) for size in layer_sizes[1:]]

    def forward(self, x):
        """Returns output of the network for one input vector (x)."""
        activation = x
        # Use sigmoid function only for hidden layers and not final layer
        for layer_id, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            logits = weights @ activation + biases
            if layer_id < len(self.weights) - 1:
                activation = sigmoid(logits)
            else:
                activation = logits

        return activation
