import numpy as np

from digits.mnist import load_data
from digits.network import Network, cross_entropy

def one_hot(label):
    """One hot encoding for labels"""
    target = np.zeros((10, 1))
    target[int(label)] = 1.0
    return target

def prepare_data(path, dataset="training", one_hot_labels=True):
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
    if one_hot_labels:
        labels = [one_hot(label) for label in labels]
    return [
        (images[index].reshape(784, 1),labels[index]) for index in range(len(images))
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
    training_data = prepare_data("data/mnist.pkl.gz", "training", one_hot_labels=True)
    validation_data = prepare_data("data/mnist.pkl.gz", "validation", one_hot_labels=True)
    test_data = prepare_data("data/mnist.pkl.gz", "test", one_hot_labels=False)
    network = Network([784, 30, 10], seed=42)
    network.SGD(
        training_data,
        epochs=10,
        mini_batch_size=64,
        eta=1.0,
        validation_data=validation_data
    )
    test_accuracy = network.evaluate_accuracy(test_data)
    print(f"Test accuracy: {test_accuracy:.2%}")
if __name__ == "__main__":
    main()