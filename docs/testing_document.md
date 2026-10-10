# Testing Document

## Overview

The project is tested using `pytest`. The tests cover the numerical functions,
the neural network, the MNIST data loader, and the training helper functions. The tests use small artificial inputs whenever possible which makes them fast,
reproducible, and independent of the complete MNIST dataset.

## Test Files

```text
src/tests/
├── test_mnist.py
├── test_network.py
└── test_train.py
```

- test_network.py tests the neural-network functions and training updates.
- test_mnist.py tests loading and transforming the dataset.
- test_train.py tests the training helper functions.

## Unit Tests

### Sigmoid

The sigmoid function is defined as:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The function is tested with known values:

```text
sigmoid(0)  = 0.5
sigmoid(1)  ≈ 0.73105858
sigmoid(-1) ≈ 0.26894142
```

The result is compared with the expected result using
`numpy.testing.assert_allclose`.

The function is also tested with extreme values:

```text
-1000, 0, 1000
```

These tests verify that the function does not produce `NaN` or infinity. They
also verify that sigmoid outputs remain within the interval $[0,1]$.

### Softmax

Softmax is tested using the logits:

```text
[1, 2, 3]
```

The expected probabilities are approximately:

```text
[0.09003057, 0.24472847, 0.66524096]
```

The tests verify that:

- the probabilities have the expected values;
- all probabilities are between `0` and `1`;
- the probabilities sum to `1`;
- extreme logits such as `[-1000, 0, 1000]` do not cause numerical overflow.

### Cross-Entropy

Cross-entropy is tested using the logits:

```text
[1, 2, 3]
```

and the one-hot target:

```text
[0, 0, 1]
```

The loss is calculated using:

$$
C=-\log(p_{\text{correct}})
$$

The function is also tested with extreme logits to verify that it returns a
finite value without numerical overflow.

### Forward Pass

A small network with the architecture

```text
2 -> 3 -> 2
```

is used to test the forward pass.

The test checks that the output has shape:

```text
(2, 1)
```

This verifies that the matrix dimensions and layer connections are correct.

### Backpropagation Gradient Shapes

A small network with the architecture

```text
3 -> 4 -> 2
```

is used to test backpropagation.

The tests verify that:

- one gradient is returned for every weight matrix;
- one gradient is returned for every bias vector;
- every gradient has the same shape as its parameter;
- all gradient values are finite.

### Independent Backpropagation Gradient Test

The backpropagation gradients are compared with gradients calculated using the
central finite-difference formula:

$$
\frac{\partial C}{\partial \theta}
\approx
\frac{C(\theta+\varepsilon)-C(\theta-\varepsilon)}
{2\varepsilon}
$$

where $\varepsilon=10^{-5}$.

The test changes every weight and bias separately. It calculates the loss after
increasing and decreasing the parameter, then compares the numerical gradient
with the gradient returned by backpropagation.

This is an independent test because it calculates the derivative from the
definition of a derivative instead of using backpropagation.

This test can detect:

- a missing sigmoid derivative in a hidden layer;
- an incorrect gradient sign;
- an incorrect matrix transpose;
- an incorrect layer index;
- an incorrect weight gradient;
- an incorrect bias gradient;
- using the wrong output-layer error;
- applying sigmoid incorrectly to the output layer.

### Mini-Batch Gradient Averaging

The test compares two identical networks.

The first network is updated with one copy of a training example. The second
network is updated with four identical copies of the same example.

The expected updates are equal because:

(1/4) × (∇C + ∇C + ∇C + ∇C) = ∇C

The final weights and biases of both networks are compared.

This test detects whether the gradients are divided by the mini-batch size. Without
the division, the update for the four-example batch would be four times larger.
### Parameter Updates and Training

The tests verify that:

- all layers change after a mini-batch update;
- stochastic gradient descent changes the weights;
- the network can overfit a small artificial dataset;
- the training loss decreases after repeated updates.

These tests check the general training behavior. The finite-difference test is
used to verify the mathematical correctness of the gradients.

### MNIST Data Loading

`test_mnist.py` creates a small artificial compressed pickle file with the same
structure as the MNIST file.

The tests verify that:

- training, validation, and test data are loaded;
- image and label arrays are returned correctly;
- a training batch has the requested size;
- images are transposed into column format;
- images and labels remain correctly aligned.

The artificial dataset is used instead of the full MNIST dataset so that the
tests remain fast and independent of the real data file.

### Training Helper Functions

`test_train.py` tests:

- one-hot encoding;
- preparation of training data;
- preparation of validation data;
- preparation of test data;
- rejection of an unknown dataset name;
- classification accuracy;
- average loss.

Dummy data and a dummy network with predefined predictions are used so that the
expected results are deterministic.

## Test Inputs

The tests use:

- ordinary positive and negative floating-point values;
- zero;
- extreme values such as `-1000` and `1000`;
- one-hot target vectors;
- small input vectors;
- small weight matrices and bias vectors;
- artificial compressed pickle files;
- repeated examples in mini-batches;
- artificial datasets with known labels.

## Reproducing the Tests

Install the project dependencies:

```bash
poetry install
```

Run all tests from the project root:

```bash
poetry run pytest
```

Run the network tests:

```bash
poetry run pytest src/tests/test_network.py
```

Run the MNIST loader tests:

```bash
poetry run pytest src/tests/test_mnist.py
```

Run the training helper tests:

```bash
poetry run pytest src/tests/test_train.py
```

Run the independent gradient test:

```bash
poetry run pytest src/tests/test_network.py -k finite_difference
```

Run the mini-batch averaging test:

```bash
poetry run pytest src/tests/test_network.py -k averages_gradients
```
## Generating a Coverage Report

`pytest-cov` is included as a development dependency and is installed automatically
with:

```bash
poetry install
```

Generate a terminal coverage report from the project root:
```bash
poetry run pytest --cov=src/digits --cov-report=term-missing
```

## Test Coverage
The report showed:
```text
Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
src/digits/__init__.py       0      0   100%
src/digits/mnist.py         15      0   100%
src/digits/network.py      105     20    81%
src/digits/train.py         44      8    82%
------------------------------------------------------
TOTAL                      164     28    83%
```
In total, 136 of 164 executable statements were covered by the tests. The total
coverage was therefore 83%. The uncovered lines were mainly related to training
and evaluation branches that were not exercised by the current unit tests.
## Empirical Results

The training program prints validation loss and validation accuracy after each
epoch:

```text
Epoch 1: validation loss=..., validation accuracy=...%
Epoch 2: validation loss=..., validation accuracy=...%
```

The results can be plotted with the epoch number on the horizontal axis. Useful
plots include:

- training loss by epoch;
- validation loss by epoch;
- training accuracy by epoch;
- validation accuracy by epoch.

The current implementation reports these values in the terminal. Graphs can be
added later by storing the values in lists and plotting them with Matplotlib.

## Limitations of the Tests

The tests use small artificial datasets, so they do not guarantee a specific
accuracy on the complete MNIST dataset.

The full training result depends on the network architecture, random
initialization, learning rate, mini-batch size, and number of epochs.

The finite-difference gradient test is more computationally expensive because it
calculates the loss many times. Therefore, it is run using a small network.