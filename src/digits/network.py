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

def evaluate(self, test_data):
    """Returns how many correctly classified examples there are"""
    correct = 0
    for x, label in test_data:
        pred = np.argmax(self.forward(x))
        correct += int(pred == label)
    return correct

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
        """Returns gradients for one training example"""
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

        # Caclulating hidden-layer errors
        for layer_id in range(2, len(self.weights) + 1):
            z = weighted_inputs[-layer_id]
            hidden_activation = activations[-layer_id]
            sigmoid_derivative = hidden_activation * (1.0 - hidden_activation)
            delta = (self.weights[-layer_id + 1].T @ delta * sigmoid_derivative)
            nabla_b[-layer_id] = delta
            nabla_w[-layer_id] = (delta @ activations[-layer_id - 1].T)

        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        """Update weights (parameters) using one mini-batch"""
        nabla_w = [np.zeros_like(weight) for weight in self.weights]
        nabla_b = [np.zeros_like(bias) for bias in self.biases]
        for x, y in mini_batch:
            delta_b, delta_w = self.backpropagation(x, y)
            for index in range(len(nabla_b)):
                nabla_b[index] += delta_b[index]
                nabla_w[index] += delta_w[index]
        batch_size = len(mini_batch)
        for id in range(len(self.weights)):
            self.weights[id] -= eta * nabla_w[id] / batch_size
            self.biases[id] -= eta * nabla_b[id] / batch_size

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Network training using mini-batch SGD"""
        training_data = list(training_data)
        for epoch in range(epochs):
            shuffled_data = [training_data[id] for id in self.rng.permutation(len(training_data))]
            mini_batches = [shuffled_data[start:start + mini_batch_size] for start in range(0, len(shuffled_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")

