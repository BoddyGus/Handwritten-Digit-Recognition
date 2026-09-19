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
    np.testing.assert_allclose(result, expected)

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
    np.testing.assert_allclose(result, expected)

def test_softmax_numerical_stability():
    logits = np.array([[-1000.0], [0.0], [1000.0]])
    result = softmax(logits)
    expected = np.array([[0.0], [0.0], [1.0]])
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected, atol=1e-12)

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

def test_forward_output_shape():
    network = Network([2, 3, 2], seed=42)
    input_vector = np.array([[0.4], [0.9]])
    result = network.forward(input_vector)
    assert result.shape == (2, 1)

