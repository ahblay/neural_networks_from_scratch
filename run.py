from layer import *
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# generate dataset
# dataset contains 2d coordinates of points
X, y = spiral_data(samples=100, classes=3)

# create layer with 2 inputs and 3 neurons
dense_1 = Layer_Dense(2, 3)

# create second dense layer (output layer) with 3 inputs and 3 neurons
dense_2 = Layer_Dense(3, 3)

# create activation function to be applied to first hidden layer
activation_1 = Activation_ReLU()

# create activation function to be applied to the output layer to normalize outputs
activation_2 = Activation_Softmax()

# create loss function to to applied to neural network output to determine accuracy
loss_function = Loss_CategoricalCrossEntropy()

# pass input into layer and calculate outputs based on random assignment of weights and biases of 0
dense_1.forward(X)

# apply activation function to dense_1 output
activation_1.forward(dense_1.output)

# feed output of first layer into the output layer
dense_2.forward(activation_1.output)

# apply the softmax function to normalize outputs
activation_2.forward(dense_2.output)

# determine the average loss on neural network output
loss = loss_function.calculate(activation_2.output, y)

# print the first 5 samples of the output
# output is a list of lists of size 3 (since dense_2 contains 3 neurons)
print(dense_2.output[:5])
print(activation_2.output[:5])
print(loss)
print(get_accuracy(activation_2.output, y))