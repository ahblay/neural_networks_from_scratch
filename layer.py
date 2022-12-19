import numpy as np
import random
from pprint import pprint as pp
import nnfs

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        Initializes a hidden layer of neurons.

        Args:
            n_inputs (int): The number of inputs k supplied to hidden layer.
            n_neurons (int): The number of neurons n in hidden layer.

        Returns:
            self.weights (2d array): A (k,n) matrix representing the weights for each input for each neuron
            self.biases (2d array): A (1,n) matrix representing the biases for each neuron
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Calculates the output of hidden layer of neurons.

        Args:
            inputs (2d array): An (x,k) matrix. A batch consists of x input vectors with k values.

        Returns:
            self.output (2d array): An (x,n) matrix. A batch of x outputs from hidden layer.
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        # assigns inputs to a class variable for reference when calculating d/dw during backpropogation
        self.inputs = inputs

    def backward(self, dvalues):
        """
        Performs a backward pass through dense layer.

        Args:
            dvalues (2d array): Passed in partials w.r.t. inputs from activation function. Each row is a sample of
                inputs and columns are partials w.r.t. z_0, z_1, ... z_n, where z_i is the ith input of the neural
                network.

        Returns:
            self.dweights (2d array):
            self.dbiases (2d array):
            self.dinputs (2d array):
        """


class Activation_ReLU:
    def forward(self, inputs):
        """
        Applies the ReLU activation function to a matrix of values.

        Args:
            inputs (2d array): For each value x in array, take max(0, x).

        Returns:
            self.output (2d array): Input matrix with all negative values replaced by 0.
        """
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        """
        Applies the softmax activation function to a matrix of values.

        Args:
            inputs (2d array): Layer inputs.

        Returns:
            self.output (2d array): Matrix where each row corresponds to normalized outputs of the neural network.
        """

        # gets exponentiated values of inputs
        # for each row, the largest value is mapped down to 0
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # gets normalized values for each number in inputs
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        """
        Calculates the loss (accuracy) of predictions from neural network output.

        Args:
            output (2d array): The output of the model.
            y (1d/2d array): The correct classification results.

        Returns:
            data_loss (float): Mean loss over all samples in input data.
        """

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        """
        Calculates the categorical cross entropy (log) loss of predictions from neural network output.

        Args:
            y_pred (2d array): The output of the model.
            y_true (1d/2d array): The correct classification results.

        Returns:
            negative_log_likelihoods (list): List of negative logs of confidence results. Complete confidence returns
                ~log(1) = 0, zero confidence returns ~log(0) = -1e-7. Predicted values clipped to avoid log(0) = -inf.
        """
        samples = len(y_pred)

        # clip predicted values by small amount to avoid taking log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7)

        # handle case where correct classification results are stored in a 1-dim list of indices
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # handle case where correct classification results are "1-hot" (i.e. incidence vectors)
        elif (y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # take negative logarithm of predicted confidence for each sample
        # maps confidence to range [inf, 0], where low confidence is heavily penalized
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


def get_accuracy(results, y):
    """
    Calculates percentage accuracy of predictions from neural network output.

    Args:
        results (2d array): The output of the model.
        y (1d/2d array): The correct classification results.

    Returns:
        accuracy (float): Percentage of predictions that are correct.
    """

    predictions = np.argmax(results, axis=1)

    # handles case where correct classification results are one-hot
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    # calculates percentage of predictions that are correct
    accuracy = np.mean(predictions==y)
    return accuracy



'''
input_size = 4
layer_size = 3
num_inputs = 10

inputs = [
    [random.randint(1, 10) for _ in range(input_size)] for _ in range(num_inputs)
    ]
weights = [
    [random.uniform(0, 1) for _ in range(input_size)] for _ in range(layer_size)
]
biases = [random.randint(1, 5) for _ in range(layer_size)]


def get_layer_output(inputs, weights, biases):
    Calculates the output of a hidden layer of neurons.

    Args:
        inputs (2d array): A batch of output values from neurons in the previous layer
        weights (2d array): A nested list of weights for each neuron in current layer
        biases (list): A list of biases for each neuron in current layer

    Returns:
        output (2d array): A nested list of batched outputs for each neuron in current layer
    
    return np.dot(inputs, np.array(weights).T) + biases

pp(inputs)
pp(weights)
pp(biases)

weights_2 = [
    [0.1, 0.2, 0.5],
    [0.55, 0.12, 0.5],
    [0.7, 0.8, 0.8]
]
biases_2 = [6, 3, 9]

layer_1_output = get_layer_output(inputs, weights, biases)
layer_2_output = get_layer_output(layer_1_output, weights_2, biases_2)

print(layer_2_output)
'''