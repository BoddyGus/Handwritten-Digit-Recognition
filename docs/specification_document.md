# Specification Document

This specification document contains the description of my project for the University of Helsinki course Algorithms and AI Project. I am studying in the Bachelor's Programme in Computer Science (TKT).

## Topic and Implementation

The topic that I have chosen for my project is Handwritten Digit Recognition. I make use of the MNIST dataset, which contains images in grayscale format of digits from 0 to 9 (handwritten). Each of the images is represented by 784 numerical pixel values (28 x 28 pixels).

The project is implemented in Python. I am also proficient in C++. I will make use of NumPy in order to implement the main machine learning algorithms and all the work with numbers myself. I implemented a feed-forward neural network and the backpropagation algorithm used to train it.

To manage the Python project, dependencies and virtual environment, I will use Poetry (recommendedation from course materials). The file `pyproject.toml` will contain the defined dependencies, and Poetry will create a lock file which would contain exact dependency versions. In order to build the project as a Python package one would have to run `poetry build`.

## The Problem


The problem my algorithm should solve is a handwritten digit classification problem involving an image into one of the classes:
0, 1, 2, 3, ..., 9.

The digits can look very different in each person's handwriting even though they are the same. A digit 1 maybe may be written many different ways using only one horizontal line, or may have a base and the diagonal line. The neural network should learn the patterns from labeled training images (image and its correct label/class) and using them predict the class of images that it has not seen yet. Main focus of the project will be mainly on neural-network training algorithm implementation and its understanding.


## Inputs

The program will receive the MNIST dataset as input, which consists of:

1. grayscale images of handwritten digits;
2. labels identifying the correct digit for each of the images;
3. separate training, validation, and test data.

As mentioned before, each image is represented by 784 numerical pixel values (28 X 28 pixels). The dataset used by this project already contains pixel values scaled approximately to the interval [0,1].

The labels are integers in between 0 and 9. During the training the labels can be represented using one-hot encoding.

The program will receive configuration parameters such as the network architecture, including the number of hidden layers and neurons in each layer, the learning rate, the random seed, the batch size, and the number of training epochs.

Training images and labels are used to calculate predictions and losses. The backpropagation algorithm calculates gradients, and mini-batch gradient descent uses them to update the weights and biases. The validation data is used to monitor the model during training, while the test data is not used during training and is reserved for the final evaluation.

## Algorithms and Data Structures

The main algorithms and methods are the following:

1. Load and preprocess the MNIST dataset;
2. Normalization of pixel values;
3. Initialization of neural-network weights and biases;
4. Forward pass;
5. Softmax output and cross-entropy loss calculations;
6. Backpropagation;
7. Gradient descent (mini-batch version SGD);
8. Classification and accuracy calculation;
9. Evaluation

The implemented network consists of an input layer with 784 values, one hidden layer with 30 neurons, and an output layer with 10 values. (probabilities for each digit class 0-9).

The sigmoid activation function is used in the hidden layer. The output layer is linear and produces logits. The softmax function converts these logits into class probabilities. The cross-entropy loss will measure how different the predictions are from the correct labels.

Backpropagation uses the chain rule of differentiation to calculate the gradients of the loss with respect to each weight and bias. Mini-batch stochastic gradient descent then updates the parameters using the calculated gradients.


The main data structures are going to be NumPy arrays. Matrices for the image data and the weights of each neural-network layer. Then vectors for biases of each layer and arrays for activations and loss gradients. The training data is shuffled and divided into mini-batches, and the parameters are updated after processing each mini-batch.

## Time and Space Complexity

Let us define the following variables:

1. $N$ is the number of training images.
2. $K=784$ is the number of input values per image.
3. $L$ is the number of neurons in the hidden layer.
4. $B$ is the mini-batch size.
5. $E$ is the number of training epochs.
6. $M$ is the number of images used for evaluation.

### Time Complexity

For the implemented network, forward propagation for one image has time
complexity

$$
O(KL + L \cdot 10).
$$

The first term comes from the matrix multiplication between the input layer and
the hidden layer. The second term comes from the matrix multiplication between
the hidden layer and the output layer.

Backpropagation performs matrix multiplications of the same dimensions in reverse
order. Therefore, its time complexity for one image is also

$$
O(KL + L \cdot 10).
$$

The sigmoid function applies one operation to each input value. For $n$ input
values, its time complexity is

$$
O(n).
$$

The softmax and cross-entropy functions process all 10 output values. Their time
complexity is

$$
O(10),
$$

which is $O(C)$ if $C$ denotes the number of output classes.

The method `update_mini_batch` calculates gradients separately for every example
in a mini-batch. Therefore, updating one mini-batch has time complexity

$$
O(B(KL + L \cdot 10)).
$$

All $N$ training images are processed during one epoch. Therefore, training for
$E$ epochs has time complexity

$$
O(EN(KL + L \cdot 10)).
$$

Evaluating the network on $M$ images requires one forward pass for each image.
Therefore, evaluation has time complexity

$$
O(M(KL + L \cdot 10)).
$$

The same complexity applies when calculating the average loss or accuracy,
because both operations require a forward pass for every evaluated image.

Loading and preprocessing $N$ images with $K$ pixel values each has time
complexity

$$
O(NK).
$$

### Space Complexity

The weights between the input and hidden layers contain $KL$ values. The weights
between the hidden and output layers contain $L \cdot 10$ values. Therefore, the
space complexity of the network parameters is

$$
O(KL + L \cdot 10).
$$

The biases require $O(L+10)$ additional space, which is included in the previous
bound.

If all $N$ training images are stored in memory, the dataset requires

$$
O(NK)
$$

space.

During backpropagation, the implementation stores activations, weighted inputs,
and gradient arrays. These arrays depend on the network size and do not change
the dominant dataset term when $N$ is large.

Therefore, the total space complexity is approximately

$$
O(NK + KL + L \cdot 10).
$$

## Sources that will be used

- [MNIST database](http://yann.lecun.com/exdb/mnist/)
- [MNIST database, Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)
- [Backpropagation, Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)
- [Gradient descent, Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [Neural network, Wikipedia](https://en.wikipedia.org/wiki/Neural_network)
- [Michael Nilsen: Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)
- [NumPy documentation](https://numpy.org/doc/stable/)



## Core of the Project

The core of the project is to implement and train a simple feed-forward neural network that is able to classify handwritten digits from images. The main algorithms are forward propagation, backpropagation, and mini-batch stochastic gradient descent. The process consists of:
1. forward propagation (to get a prediction);
2. calculating the cross-entropy loss by comparing the predicted probabilities with the correct one-hot label
3. backpropagation to calculate gradients with respect to the weights and biases
4. gradient descent to update the weights and biases and reduce the loss

This process will be repeated for several epochs. After training, the network is evaluated on the separate test images and their labels.

Displaying results and loading the dataset are supporting parts of the project. The main focus is the implementation and understanding of forward propagation, backpropagation, loss calculation, and parameter updates.