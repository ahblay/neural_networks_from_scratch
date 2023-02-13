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

The only requirements for this project are the Numpy library and a library created by the authors to generate sample datasets. They can be installed by running the following commands:

`pip install numpy`
`pip install nnfs`

## How to Use

To use the simple neural network, follow these steps:

1. Import the all classes from `layer.py`
2. Instantiate `Layer_Dense` objects for each layer with the desired number of inputs and outputs
3. Instantiate activation functions for the hidden and output layers, and a loss function to determine how to adjust weights/biases for learning
4. Use the `predict` method to make predictions on new data

An example of how to use the network is provided in the `run.py` file.

## Backpropagation

Backpropagation is a method for updating the weights and biases in the network to minimize the loss. It involves propagating the errors from the output layer to the input layer and using gradient descent to update the weights and biases in the network. The gradients are calculated using the chain rule of differentiation.

## Conclusion

This project builds a simple neural network from scratch using Numpy library and follows the instructions from the book "Neural Networks from Scratch in Python" by Harrison Kinsley and Daniel Kukiela. The network can be trained on data to make predictions and can be further expanded to build more complex networks.
