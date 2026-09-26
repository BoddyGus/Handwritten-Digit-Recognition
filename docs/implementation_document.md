# Implementation document

## Overview

In the projectIn the project, I implemented a feed-forward neural network for handwritten digit classification using Python and NumPy. I wasn't using any machine learning framework for the model itself; in other words, I implemented the neural network from scratch. The implementation includes the forward pass, activation functions, softmax, cross-entropy loss, backpropagation, mini-batch gradient descent and stochastic gradient descent.

The project's structure consists of modules for the neural network, MNIST data loading as well as training. The main idea was to focus on the mathematical operations of a neural network and algorithms themselves rather than using high-level machine learning libraries.

The file network.py contains the main implementation. The neural network is represented in the Network class in the network.py file. The class is responsible for initializing the parameters, performing forward propagation, calculating gradients using backpropagation, and updating the network parameters during training. Loading of the data and training functionality are implemented in the mnist.py and train.py files, respectively.
