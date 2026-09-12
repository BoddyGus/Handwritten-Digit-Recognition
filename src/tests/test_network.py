import numpy as np
import pytest

from digits.network import Network, sigmoid


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0.5),
        (1.0, 1.0 / (1.0 + np.exp(-1.0))),
        (-1.0, 1.0 / (1.0 + np.exp(1.0))),
    ],
)

def test_sigmoid_compared_to_known_values(value, expected):
    result = sigmoid(np.array([value]))
    assert result[0] == pytest.approx(expected)

def test_forward_output_shape():
    network = Network([2, 3, 2], seed=42)
    input_vector = np.array([[0.4], [0.9]])
    result = network.forward(input_vector)
    assert result.shape == (2, 1)

