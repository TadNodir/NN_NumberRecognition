import numpy as np


class Perceptron:

    def __init__(self):
        self.synaptic_weights = np.random.rand(3, 1) - 1

    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # training loop
    def train(self, inputs, target, iterations):
        for _ in range(iterations):
            output = self.think(inputs)
            e_out = target - output
            temp = e_out * self.sigmoid_derivative(output)
            delta_w = np.dot(inputs.T, temp)
            self.synaptic_weights += delta_w

    # one calculation step of the perceptron
    def think(self, inputs):
        product = np.dot(inputs, self.synaptic_weights)
        sigmoid_fnk = self.sigmoid(product)
        return sigmoid_fnk


if __name__ == "__main__":
    train_inputs = np.array([[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 0, 1]]).T
    targets = np.array([[0, 1, 1, 0]]).T
    p = Perceptron()
    print("Weights before: \n", p.synaptic_weights)
    p.train(train_inputs, targets, 10000)
    print("Weights after: \n", p.synaptic_weights)

    print()
    test_input = np.array([0, 0, 1])
    predict = p.think(test_input)
    print("Prediction for [0, 0, 1]: ", predict)
    test_input = np.array([1, 1, 1])
    predict = p.think(test_input)
    print("Prediction for [1, 1, 1]: ", predict)
    test_input = np.array([1, 0, 0])
    predict = p.think(test_input)
    print("Prediction for [1, 0, 0]: ", predict)
    test_input = np.array([0, 1, 1])
    predict = p.think(test_input)
    print("Prediction for [0, 1, 1]: ", predict)
    test_input = np.array([1, 1, 0])
    predict = p.think(test_input)
    print("Prediction for [1, 1, 0]: ", predict)
    test_input = np.array([0, 0, 0])
    predict = p.think(test_input)
    print("Prediction for [0, 0, 0]: ", predict)
    test_input = np.array([0, 1, 0])
    predict = p.think(test_input)
    print("Prediction for [0, 1, 0]: ", predict)
