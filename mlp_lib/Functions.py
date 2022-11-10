import math

import numpy as np


class Activation:
    sigmoid = ("sigmoid", "Sigmoid", "SIGMOID", "sig")
    relu = ("relu", "Relu", "RELU", "ReLU")
    leaky_relu = ("leaky_relu", "Leaky_ReLU", "Leaky ReLU", "leaky relu", "LEAKY_RELU")
    tanh = ("tanh", "Tanh", "TANH")
    elu = ("elu", "Elu", "ELU")


class Loss:
    empirical = ("empirical", "Empirical", "EMPIRICAL")
    mse = ("mse", "MSE", "Mse", "mean_squared_error", "Mean Squared Error")


class Optimizer:
    sgd = ("SGD", "sgd", "Sgd", "stochastic_gradient_descent", "Stochastic Gradient Descent")


def activation(type: str):
    if type in Activation.sigmoid:
        return lambda input: 1 / (1 + math.e ** (-input))
    if type in Activation.relu:
        return lambda input: max(0, input)
    if type in Activation.leaky_relu:
        return lambda input: input if input > 0 else input * 0.01
    if type in Activation.tanh:
        return lambda input: math.tanh(input)
    if type in Activation.elu:
        return lambda input: input if input > 0 else math.e ** input - 1


def loss(type: str):
    if type in Loss.empirical:
        return lambda logits, labels: np.average(np.subtract(logits, labels))
    if type in Loss.mse:
        return lambda logits, labels: 1 / len(logits) * sum((np.subtract(labels, logits)) ** 2)
    # if type == "binary_crossentropy":


def optimizer(type: str):
    if type == "sgd":
        pass
