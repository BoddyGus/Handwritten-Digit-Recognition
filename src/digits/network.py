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

def cross_entropy(logits, target):
    """A softmax cross-entropy function (numerically stable version) for one example """
    logits = np.asarray(logits, dtype=float)
    target = np.asarray(target, dtype=float)
    maximum = np.max(logits, axis=0, keepdims=True)
    shift_logits = logits - maximum
    log_sum_exp = maximum + np.log(np.sum(np.exp(shift_logits), axis=0, keepdims=True))
    loss = log_sum_exp - np.sum(target * logits, axis=0, keepdims=True)
    return float(np.squeeze(loss))

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

    def backpropagation(self, x, y):
        """Returs gradients for one training example"""
        nabla_w = [np.zeros_like(weight) for weight in self.weights]
        nabla_b = [np.zeros_like(bias) for bias in self.biases]
        activation = x
        activations = [activation]
        weighted_inputs = []

        # Forward pass
        for layer_id, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = weight @ activation + bias
            weighted_inputs.append(z)
            if layer_id < len(self.weights) - 1:
                activation = sigmoid(z)
            else:
                activation = z
            activations.append(activation)

        # Error of an output for softmax + cross-entropy
        probabilities = softmax(activations[-1])
        delta = probabilities - y

        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T



