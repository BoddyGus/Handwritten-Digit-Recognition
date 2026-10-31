# User Guide

## Requirements

- Python 3.12 or newer
- Poetry

## Installing the Project

Clone the repository and enter its directory:

```bash
git clone https://github.com/BoddyGus/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

Install the project dependencies:

```bash
poetry install
```

## MNIST Dataset

The program expects the file `data/mnist.pkl.gz`. It contains the MNIST
dataset divided into training, validation, and test sets. The file is loaded by
`digits.mnist.load_data`.

Download the dataset from the project source or place an equivalent MNIST
pickle file at:

```text
data/mnist.pkl.gz
```

The file must contain three (images, labels) pairs. Images must contain
784-pixel grayscale vectors, and labels must contain digit values from 0 to 9.

## Training Parameters

The training parameters are currently constants in `src/digits/train.py`.
They are not read from command-line arguments or a configuration file.

The current values are:

- network architecture: `784 -> 30 -> 10`;
- number of epochs: `10`;
- mini-batch size: `64`;
- learning rate: `1.0`;
- random seed: `42`.

To change these values, edit the corresponding arguments in `main()` in
`src/digits/train.py`.

## Training Output

Run the training program with:

```bash
poetry run python -m digits.train
```

The program prints validation loss and validation accuracy after every epoch.
After training, it prints the test accuracy:

Epoch 1: validation loss=..., validation accuracy=...%
...
Test accuracy: ...%

Training on the complete MNIST dataset can take from several seconds to several
minutes depending on the computer. The exact time depends on the number of
epochs, mini-batch size, and network architecture.

The training program receives its input from `data/mnist.pkl.gz`. It does not
currently accept command-line arguments.

The neural-network functions operate on NumPy column vectors. Input images have
shape `(784, 1)`. Targets are either one-hot vectors with shape `(10, 1)` or
integer labels, depending on the function being used.

## Running Tests

Run all tests from the project root:

```bash
poetry run pytest
```

Run a specific test file by replacing <test-file> with its path:

```bash
poetry run pytest src/tests/<test-file>
```

For example:
```bash
poetry run pytest src/tests/test_network.py
```

## Running Training

Run the training from the project root:
```bash
poetry run python -m digits.train
```
