from numpy import exp, array, random, dot, vstack, mean, power, asarray
import numpy as np

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.layers = 1
        self.weights = {}
        self.adjustments = {}

    def add_layer(self, shape):
        self.weights[self.layers] = vstack(2 * (random.random(shape)) - 1, 2 * (random.random((1, shape[1])) - 1))
        self.adjustments[self.layers] = np.zeros(shape)
        self.layers += 1

    def __sigmoid(self, x): 
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __sum_squared_error(self, outputs, targets):
        return 0.5 * mean(np.sum(power(outputs - targets, 2), axis=1))

    def think(self, data):
        for layer in range(1, len(self.layers)):
            data = self.__sigmoid(dot(data, self.weights[layer]))
        return data
    def __forward_propigate(self, data):

    def __back_propigate(self, output, target):

    def train(self, training_data, training_lables, itterations, training_speed:)


if __name__ == '__main__':

    training_data = np.asarray([[1,1,1], [0,0,0], [1,1,0], [1,0,0], [0,1,1], [0,0,1], [0,1,0]])
    training_lables = np.asarray([[0], [0], [0], [1], [0], [1], [1]])

    neuralNet = NeuralNetwork()
    neuralNet.__init__()
    neuralNet.add_layer((3,9))
    neuralNet.add_layer((9,1))

    neuralNet.think(np.asarray([1,0,1]))