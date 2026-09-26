import numpy as np

from digits.mnist import load_data
from digits.network import Network, cross_entropy

def one_hot(label):
    """One hot encoding for labels"""
    target = np.zeros((10, 1))
    target[int(label)] = 1.0
    return target

def prepare_data(path, dataset="training"):
    training_data, validation_data, test_data = load_data(path)

    if dataset == "training":
        images, labels = training_data
    elif dataset == "validation":
        images, labels = validation_data
    elif dataset == "test":
        images, labels = test_data
    else:
        raise ValueError("Unknown dataset")
    images = np.asarray(images, dtype=float)
    labels = np.asarray(labels)
    return [
        (images[index].reshape(784, 1), one_hot(labels[index]))
        for index in range(len(images))
    ]

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
    training_data = prepare_data("data/mnist.pkl.gz", "training")
    validation_data = prepare_data("data/mnist.pkl.gz", "validation")
    test_data = prepare_data("data/mnist.pkl.gz", "test")
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
        # Evaluate training data
        train_loss = avg_loss(network, training_data)
        train_accuracy = accuracy(network, training_data)
        # Evaluate validation data
        validation_loss = avg_loss(network, validation_data)
        validation_accuracy = accuracy(network, validation_data)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.2%}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.2%}"
        )
    # Final evaluation on completely unseen test data
    test_loss = avg_loss(network, test_data)
    test_accuracy = accuracy(network, test_data)
    print(
        f"\nFinal test performance: "
        f"loss={test_loss:.4f}, "
        f"accuracy={test_accuracy:.2%}"
    )
if __name__ == "__main__":
    main()