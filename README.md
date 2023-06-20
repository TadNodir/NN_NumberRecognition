# NN_NumberRecognition

This repository contains the implementation of a feed-forward neural network for recognizing handwritten numbers from 0 to 9 using the MNIST dataset. The network architecture consists of an input layer, a hidden layer, and an output layer. Each layer is composed of perceptrons interconnected with weighted connections.

## Dataset

The MNIST dataset is used for training and testing the network. It contains thousands of 28x28 grayscale images of handwritten numbers. You can download the dataset from the following links:

- [mnist_train_100.csv](link-to-mnist_train_100.csv): Reduced training dataset (recommended for development)
- [mnist_train_full.csv](link-to-mnist_train_full.csv): Full training dataset
- [mnist_test_10.csv](link-to-mnist_test_10.csv): Reduced test dataset
- [mnist_test_full.csv](link-to-mnist_test_full.csv): Full test dataset

The dataset is stored in a CSV format, where each row represents an image. The first column contains the true value or label of the image, followed by the grayscale pixel values.

## Network Architecture

The feed-forward network is designed as follows:

- Input Layer (I): Consists of 784 perceptrons, each representing a pixel of the grayscale image.
- Hidden Layer (H): Comprises 200 perceptrons.
- Output Layer (O): Contains 10 perceptrons, representing the numbers from 0 to 9.

The weighted connections between the perceptrons of the input layer and the hidden layer are modeled using a 200x784 weight matrix (W_ih). Similarly, the connections between the hidden layer and the output layer are represented by a 10x200 weight matrix (W_ho).

## Implementation

The network is implemented as a Python class named `FeedForwardNetwork`. It provides the following member functions:

- `__init__(self)`: Initializes all necessary member variables.
- `sigmoid(self, x)`: Implements the sigmoid activation function.
- `sigmoid_derivative(self, x)`: Calculates the derivative of the sigmoid function.
- `think(self, inputs)`: Performs one "thought-step" of the network for the given inputs.
- `train(self, inputs, targets, iterations)`: The training loop that executes "iterations" times to train the network. It includes backpropagation and weight adjustments.
- `__name__ == "__main__"`: The main function that creates an instance of the `FeedForwardNetwork` class, trains it using the MNIST training dataset, and tests it with a test dataset. It returns a performance rating indicating the percentage of correctly recognized images.
- Create a dataset of your own handwriting and let the network recognize it. Measure the recognition rate.

## Dependencies

The implementation requires the following dependencies:

- Python 3.x
- NumPy

Please make sure to install the required dependencies before running the code.

## Usage

To use the `FeedForwardNetwork` class, follow these steps:

1. Download the MNIST dataset from the provided links and place the CSV files in the same directory as the code files.
2. Import the `FeedForwardNetwork` class into your Python program.
3. Create an instance of the `FeedForwardNetwork` class.
4. Train the network using the `train` method by providing the MNIST training dataset and the desired number of iterations.
5. Test the network's performance using the `think` method and the MNIST test dataset.
6. Optionally, create a dataset of your own handwriting and use the `think` method to recognize the handwritten numbers.

## Results

The network's performance can be measured by the percentage of correctly recognized images
