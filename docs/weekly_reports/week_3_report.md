This week I have implemented all necessary components of the network for handwritten digit recognition, so now I have a fully functional neural network which can be trained by running the train.py file. Also I have run a code analysis with pylint and started my testing_document.

As for the code, I added the following functions: cross_entropy (numerically stable version), backpropagation, update_mini_batch and SGD. On top of that I have created and run unit tests for all of the functions in network.py. I have also created the file train.py which runs the training process of the neural network.

This week I have learned quite a lot. I got to understand much more specific information behind all of the new algorithms I have implemented this week (listed above), math behind them, why specific things work a certain way, the goal of the whole process etc.

Everything remains quite clear, nothing to pinpoint.

My next steps are first of all, implementing overall performance measurement in train.py using test_data as well as visualization of the perfomance of my neural network. Also I plan to write all unit tests information into testing_document.md file.