import numpy as np
import pytest

from digits.network import Network, sigmoid, softmax, cross_entropy


def test_sigmoid_compared_to_known_values():
    values = np.array([0.0, 1.0, -1.0])
    result = sigmoid(values)
    expected = np.array([
        0.5,
        0.73105858,
        0.26894142,
    ])
    assert result == pytest.approx(expected)

def test_sigmoid_numerical_stability():
    values = np.array([-1000.0, 0, 1000.0])
    result = sigmoid(values)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx(1.0)

def test_sigmoid_within_correct_interval():
    values = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    result = sigmoid(values)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)

def test_softmax_compared_to_known_values():
    logits = np.array([[1.0], [2.0], [3.0]])
    result = softmax(logits)
    expected = np.array([
        [0.09003057],
        [0.24472847],
        [0.66524096],
    ])
    assert result == pytest.approx(expected)

def test_softmax_numerical_stability():
    logits = np.array([[-1000.0], [0.0], [1000.0]])
    result = softmax(logits)
    expected = np.array([[0.0], [0.0], [1.0]])
    assert np.all(np.isfinite(result))
    assert result == pytest.approx(expected, abs=1e-12)

def test_softmax_within_correct_interval():
    logits = np.array([[-1000.0], [0.0], [1000.0]])
    result = softmax(logits)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)

def test_softmax_sum_of_probabilities_is_one():
    logits = np.array([[1.0], [2.0], [3.0]])
    result = softmax(logits)
    assert np.sum(result) == pytest.approx(1.0)

def test_cross_entropy_compared_to_known_values():
    logits = np.array([[1.0], [2.0], [3.0]])
    target = np.array([[0.0], [0.0], [1.0]])
    result = cross_entropy(logits, target)
    expected = -np.log(0.66524096)
    assert result == pytest.approx(expected)

def test_cross_entropy_numerical_stability():
    logits = np.array([[-1000.0], [0.0], [1000.0]])
    target = np.array([[0.0], [0.0], [1.0]])
    result = cross_entropy(logits, target)
    assert np.isfinite(result)
    assert result == pytest.approx(0.0)

def test_backpropagation_shapes_of_gradients():
    network = Network([3,3,2], seed=42)
    x = np.array([[0.1], [0.2], [0.3]])
    y = np.array([[0.0], [1.0]])
    grads_b, grads_w = network.backpropagation(x, y)
    assert len(grads_b) == len(network.biases)
    assert len(grads_w) == len(network.weights)
    for grad, bias in zip(grads_b, network.biases):
        assert grad.shape == bias.shape
        assert np.all(np.isfinite(grad))
    for grad, weight in zip(grads_w, network.weights):
        assert grad.shape == weight.shape
        assert np.all(np.isfinite(grad))

def test_all_layers_change_after_update():
    network = Network([2, 3, 2], seed=42)

    mini_batch = [
        (np.array([[0.1], [0.7]]), np.array([[1.0], [0.0]])
        ),
        (
            np.array([[0.6], [0.1]]),
            np.array([[0.0], [1.0]])
        ),
    ]
    old_weights = [weight.copy() for weight in network.weights]
    old_biases = [bias.copy() for bias in network.biases]
    network.update_mini_batch(mini_batch, eta=0.1)
    for old, new in zip(old_weights, network.weights):
        assert not np.allclose(old, new)

    for old, new in zip(old_biases, network.biases):
        assert not np.allclose(old, new)

def test_sgd_updates_weights():
    network = Network([2, 4, 2], seed=42)
    training_data = [
        (np.array([[0.2], [0.7]]), np.array([[1.0], [0.0]])),
        (np.array([[0.8], [0.1]]), np.array([[0.0], [1.0]])),
    ]
    old_weights = [weight.copy() for weight in network.weights]
    network.SGD(training_data, epochs=2, mini_batch_size=1, eta=0.1)
    assert any(not np.allclose(old, new) for old, new in zip(old_weights, network.weights))

def test_forward_output_shape():
    network = Network([2, 3, 2], seed=42)
    input_vector = np.array([[0.4], [0.9]])
    result = network.forward(input_vector)
    assert result.shape == (2, 1)

def test_network_can_overfit_small_dataset():
    network = Network([2, 4, 2], seed=42)
    training_data = [
        (np.array([[0.0], [0.0]]), np.array([[1.0], [0.0]])),
        (np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]])),
        (np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])),
        (np.array([[1.0], [1.0]]), np.array([[0.0], [1.0]])),
    ]
    for _ in range(1000):
        network.update_mini_batch(training_data, eta=1.0)

    correct = 0
    for x, target in training_data:
        prediction = np.argmax(network.forward(x))
        expected = np.argmax(target)
        correct += int(prediction == expected)
    accuracy = correct / len(training_data)
    assert accuracy == 1.0

def test_training_reduces_loss():
    network = Network([2, 4, 2], seed=42)
    training_data = [
        (np.array([[0.0], [0.0]]), np.array([[1.0], [0.0]])),
        (np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]])),
        (np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])),
        (np.array([[1.0], [1.0]]), np.array([[0.0], [1.0]])),
    ]
    def loss():
        losses = []
        for x, target in training_data:
            losses.append(cross_entropy(network.forward(x), target))

        return np.mean(losses)
    start_loss = loss()
    for _ in range(100):
        network.update_mini_batch(training_data, eta=1.0)
    end_loss = loss()

    assert end_loss < start_loss


def test_sample_order_does_not_change_predictions():
    network = Network([2, 3, 2], seed=42)
    samples = [
        np.array([[0.1], [0.2]]),
        np.array([[0.3], [0.4]]),
        np.array([[0.5], [0.6]]),
    ]
    initial_predictions = [network.forward(x) for x in samples]
    shuffled_samples = [samples[2], samples[0], samples[1]]
    shuffled_predictions = [network.forward(x) for x in shuffled_samples]
    assert shuffled_predictions[1] == pytest.approx(initial_predictions[0])
    assert shuffled_predictions[2] == pytest.approx(initial_predictions[1])
    assert shuffled_predictions[0] == pytest.approx(initial_predictions[2])