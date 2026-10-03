# Implementation document

## Overview

In the project, I implemented a feed-forward neural network for handwritten digit classification using Python and NumPy. I wasn't using any machine learning framework for the model itself; in other words, I implemented the neural network from scratch. The implementation includes the forward pass, activation functions, softmax, cross-entropy loss, backpropagation, mini-batch gradient descent and stochastic gradient descent.

The project's structure consists of modules for the neural network, MNIST data loading as well as training. The main idea was to focus on the mathematical operations of a neural network and algorithms themselves rather than using high-level machine learning libraries.

The file network.py contains the main implementation. The neural network is represented in the Network class in the network.py file. The class is responsible for initializing the parameters, performing forward propagation, calculating gradients using backpropagation, and updating the network parameters during training. Loading of the data and training functionality are implemented in the mnist.py and train.py files, respectively.

## Project Structure

```text
Handwritten-Digit-Recognition/
├── data/
│   └── mnist.pkl.gz
├── docs/
│   ├── specification_document.md
│   ├── implementation_document.md
│   ├── testing_document.md
│   ├── pylint-report.txt
│   └── weekly_reports/
├── src/
│   ├── digits/
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   └── network.py
│   │   └── train.py
│   └── tests/
│       ├── test_mnist.py
│       ├── test_network.py
│       └── test_train.py
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```
## Time and Space Complexity

## Possible Shortcomings and Improvements
- The implementation currently processes training examples individually inside each
mini-batch. Fully vectorized batch operations could improve performance by using
NumPy matrix multiplication more efficiently.
- The project does not currently include advanced optimizers such as Adam or learning-rate scheduling that could improve convergence.
- The model architecture and hyperparameters are simple and may not give the best
possible MNIST accuracy. Improvements such as testing different numbers
of hidden layers, hidden neurons, learning rates, batch sizes, and epochs would be benefitial.

## LLM usage

I have used GPT-5.5 as an assistant during my work on the project. It helped me fix some of the isolated errors in specific functions, create dummy data for some of the unit tests, and check the spelling and grammar of the porject reports.

The implementation, algorithmic decisions, testing, and final verification were done by me.


## Final Sources

- [The MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- [MNIST Database - Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)
- [Backpropagation - Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)
- [Gradient Descent - Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [Neural Network - Wikipedia](https://en.wikipedia.org/wiki/Neural_network)
- [Michael Nielsen: Neural Networks and Deep Learning, Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Michael Nielsen: How the Backpropagation Algorithm Works, Chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [pytest Documentation](https://docs.pytest.org/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Stable Softmax - Jay Mody](https://jaykmody.com/blog/stable-softmax/)
- [Optimal Way of Defining a Numerically Stable Sigmoid Function for a List in Python - Stack Overflow](https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth)
- [Writing Automated Tests for Neural Networks - Sebastian Björkqvist](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/)