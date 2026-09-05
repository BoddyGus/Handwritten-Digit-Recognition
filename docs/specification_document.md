# Specification Document

This specification document contains the description of my project for the University of Helsinki course Algorithms and AI Project. I am studying in the Bachelor's Programme in Computer Science (TKT).

## Topic and Implementation

The topic that I have chosen for my project is Handwritten Digit Recognition. I make use of the MNIST dataset, which contains images in grayscale format of digits from 0 to 9 (handwritten). Each of the images is represented by 784 numerical pixel values (28 x 28 pixels).

The project will be implemented in Python (also the only languages I am proficient are C++ and Python). I will make use of NumPy in order to implement the main machine learning algorithms and all the work with numbers myself. In particular, I plan to implement a feed-forward neural network and the backpropagation algorithm which will be used to train the network. I will run the training on the Roihu cluster since training a neural network over the whole MNIST dataset can require a lot of computation as well as well as to gain experience of taking advantage of cluster's resources.

To manage the Python project, dependencies and virtual environment, I will use Poetry (recommendedation from course materials). The file `pyproject.toml` will contain the defined dependencies, and Poetry will create a lock file which would contain exact dependency versions. In order to build the project as a Python package one would have to run `poetry build`.

## The Problem


The problem my algorithm should solve is a handwritten digit classification problem involving an image into one of the classes:
0, 1, 2, 3, ..., 9.

The digits can look very different in each person's handwriting even though they are the same. A digit 1 maybe may be written many different ways using only one horizontal line, or may have a base and the diagonal line. The neural network should learn the patterns from labeled training images (image and its correct label/class) and using them predict the class of images that it has not seen yet. Main focus of the project will be mainly on neural-network training algorithm implementation and its understanding.


## Inputs

The program will receive the MNIST dataset as input, which consists of:

1. grayscale images of handwritten digits;
2. labels identifying the correct digit for each of the images;
3. separate training and test data.

As mentioned before, each image is represented by 784 numerical pixel values (28 X 28 pixels). They could be normalized to be in between 0 and 1.

The labels are integers in between 0 and 9. During the training the labels can be represented using one-hot encoding.

Also the program will receive configuration parametrs such as the number of hidden layers, the number of neurons in each layer, the learning rate, the random seed, the batch size and the number of training epochs.

Training images and labels will be used to calculate predictions and errors. After that, backpropogation algorithm will use the losses to calculate gradients and update the weights and biases. The test portion of images and labels won't be used in training, but will be used only for evaluation accuracy of model's predictions.

## Algorithms and Data Structures

The main algorithms and methods are the following:

1. Load and preprocess the MNIST dataset;
2. Normalization of pixel values;
3. Initialization of neural-network weights and biases;
4. Forward pass;
5. Softmax output and cross-entropy loss calculations;
6. Backpropagation;
7. Gradient descent (mini-batch version SGD);
8. Classsification and accuracy calculation;
9. Evaluation

The neural network will consist of an input layer with 784 input values, two or more hidden layers and finally, an output layer with 10 values (probabilities for each digit class 0-9).

An activavtion function ReLU or GELU will be used inside the hidden layers. The output layer will have a softmax function so that its outputs could be treated as probabilities. The cross-entropy loss will measure how different the predictions are from the correct labels.

Backpropagation makes use of the chain rule of differentiation to calculate the contributions of each weight and bias to the prediction error. Then SGD (mini-batch gradient descent) will update the parameters using gradients we got.


The main data structures are going to be NumPy arrays. Matrices for the image data and the weights of each neural-network layer. Then vectors for biases of each layer and arrays for activations and loss gradients. The batches will contain subsets of the training data.

## Time and Space Complexity

Let us define some variables for simplicity:
1. $N$  - number of training images,
2. $K$ = 784 - number of input values per image ()
3. $L$ - the number of neurons in one hidden layer.


Forward propagation is done through matrix multiplications, which have a complexity of $O(KL + L \cdot 10)$ (for one image).

Backpropagation's matrix operations are similar, so it has the same complexity.


If the network is trained for $E$ epochs with the use of all $N$ training images, the total training time would be $O(E N (KL + L \cdot 10))$.

The space complexity of weights and biases is approximately $O(KL + L \cdot 10)$.

Now, if all images that are used for training are stored in memory, then the dataset would require $O(NK)$ space. So the total space complexity would be:
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

The core of the project is to implement and train a simple feed-forward neural networkk that is able to classify handwritten digits from images. The main algorithms are forward- and backpropagation, gradient descent. The process consists of:
1. forward propagation (to get a prediction);
2. comparing prediction with the correct label using loss function
3. backpropagation to calculate gradients and biases;
4. gradient descent to update gradients and biases to reduce the error

This process will be repeated for several epochs. After that trained network is tested using test images and their correct labels.

Displaying results will be part of the project. However, main focud will be on implementation and understanding the neural-network training algorithm.