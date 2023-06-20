import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


class NeuralNetwork:

    # initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes
        self.learning_rate = learningrate
        self.W_ih = np.random.rand(hiddennodes, inputnodes) - 1
        self.W_ho = np.random.rand(outputnodes, hiddennodes) - 1

    # sigmoid function
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # train the neural network
    def train(self, inputs, targets, iterations):
        for _ in range(iterations):
            for i in range(len(inputs)):
                input_data = inputs[i]
                output_data = targets[i]
                h, o = self.think(input_data)
                e_out = output_data - o
                e_hidden = np.dot(self.W_ho.T, e_out)
                tmp = e_out * self.sigmoid_derivative(o)
                delta_W_ho = np.outer(tmp, h.T)
                self.W_ho = self.W_ho + self.learning_rate * delta_W_ho
                tmp2 = e_hidden * self.sigmoid_derivative(h)
                delta_W_ih = np.outer(tmp2, input_data)
                self.W_ih = self.W_ih + self.learning_rate * delta_W_ih

    # one calculation step of the network
    def think(self, inputs):
        h = self.sigmoid(np.dot(self.W_ih, inputs))
        o = self.sigmoid(np.dot(self.W_ho, h))
        return h, o

    def print_prediction(self, inputs):
        h, o = self.think(inputs)
        predicted_number = np.argmax(o)
        print("Predicted number is: ", predicted_number)

    def test_model(self, inputs, targets):
        result_t = []
        for i in range(len(inputs)):
            input_v = inputs[i]
            target_v = targets[i]
            hidden, outputs = self.think(input_v)
            label = np.argmax(outputs)
            if label == target_v:
                result_t.append(1)
            else:
                result_t.append(0)

        return result_t


if __name__ == "__main__":
    input_nodes = 784  # 28*28 pixel
    hidden_nodes = 200  # voodoo magic number
    output_nodes = 10  # numbers from [0:9]

    learning_rate = 0.1  # feel free to play around with

    training_data_file = open("mnist_train_100.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    test_data_file = open("mnist_test_10.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # preparing data to train
    training_inputs = []
    training_targets = []
    test_inputs = []
    test_targets = []
    for val in training_data_list:
        elements = val.split(',')
        target_values = np.zeros(output_nodes) + 0.01
        target_values[int(elements[0])] = 0.99
        input_values = (np.asfarray(elements[1:]) / 255 * 0.98) + 0.01
        training_inputs.append(input_values)
        training_targets.append(target_values)

    for val in test_data_list:
        elements = val.split(',')
        t_targets = float(elements[0])
        t_inputs = (np.asfarray(elements[1:]) / 255 * 0.98) + 0.01
        test_targets.append(t_targets)
        test_inputs.append(t_inputs)

    test_inputs = np.array(test_inputs)
    test_targets = np.array(test_targets)

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    n.train(training_inputs, training_targets, 100)

    result = n.test_model(test_inputs, test_targets)
    accuracy = sum(result) / len(result) * 100
    print("Accuracy: ", accuracy)

    guess_number = test_inputs[1]
    n.print_prediction(guess_number)
    print("Plotting image: ")
    image_array = np.asfarray(guess_number).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()
