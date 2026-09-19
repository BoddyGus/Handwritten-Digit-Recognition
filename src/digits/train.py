import numpy as np

from digits.mnist import load_data
from digits.network import Network, cross_entropy

def one_hot(label):
    """One hot encoding for labels"""
    target = np.zeros((10, 1))
    target[int(label)] = 1.0
    return target

def prepare_data_for_training(path):
    (images, labels), _, _ = load_data(path)
    images = np.asarray(images, dtype=float)
    labels = np.asarray(labels)
    return [(images[index].reshape(784, 1), one_hot(labels[index])) for index in range(len(images))]

def accuracy(network, data):
    correct = 0
    for image, target in data:
        pred = np.argmax(network.forward(image))
        correct_label = np.argmax(target) # returns the same same class as np.argmax(softmax(logits))

        correct += int(pred == correct_label)
    return correct / len(data)

def avg_loss(network, data):
    losses = []
    for image, target in data:
        logits = network.forward(image)
        losses.append(cross_entropy(logits, target))
    return float(np.mean(losses))



def main():
    training_data = prepare_data_for_training("data/mnist.pkl.gz",)
    network = Network([784, 30, 10], seed=42)
    epochs = 10
    mini_batch_size = 64
    lr = 1.0

    for epoch in range(epochs):
        perm = network.rng.permutation(len(training_data))
        shuffled_data = [training_data[index] for index in perm]
        for start in range(0, len(shuffled_data), mini_batch_size):
            mini_batch = shuffled_data[start:start + mini_batch_size]
            network.update_mini_batch(mini_batch,lr)
        epoch_loss = avg_loss(network, training_data)
        epoch_accuracy = accuracy(network, training_data)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={epoch_loss:.4f}, "
            f"accuracy={epoch_accuracy:.2%}"
        )

if __name__ == "__main__":
    main()