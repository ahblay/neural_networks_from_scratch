# Simple Neural Network from Scratch in Numpy

This project builds a simple neural network from scratch using Numpy library. The instructions for building the network are taken from the book "Neural Networks from Scratch in Python" by Harrison Kinsley and Daniel Kukiela.

## Features

The basic features of the simple neural network include:

- Initialization of the network weights and biases
- Feedforward of inputs through the network to make predictions
- Calculation of loss based on predicted output and actual target
- Backpropagation of errors to update the weights and biases in the network
- Training of the network using stochastic gradient descent

## Requirements

The only requirement for this project is the Numpy library. It can be installed by running the following command:

`pip install numpy`

## How to Use

To use the simple neural network, follow these steps:

1. Import the `NeuralNetwork` class from the `neural_network.py` file
2. Initialize a network object with the desired number of inputs, hidden layers, and outputs
3. Train the network using the `fit` method by passing in the inputs and targets for the training data
4. Use the `predict` method to make predictions on new data

An example of how to use the network is provided in the `example.py` file.

## Backpropagation

Backpropagation is a method for updating the weights and biases in the network to minimize the loss. It involves propagating the errors from the output layer to the input layer and using gradient descent to update the weights and biases in the network. The gradients are calculated using the chain rule of differentiation.

## Conclusion

This project builds a simple neural network from scratch using Numpy library and follows the instructions from the book "Neural Networks from Scratch in Python" by Harrison Kinsley and Daniel Kukiela. The network can be trained on data to make predictions and can be further expanded to build more complex networks.
