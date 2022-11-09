import math

import numpy as np


def activation(type: str):
    if type == "sigmoid":
        return lambda input: 1 / (1 + math.e ** (-input))
    if type == "relu":
        return lambda input: max(0, input)
    if type == "leaky_relu":
        return lambda input: input if input > 0 else input * 0.01
    if type == "tanh":
        return lambda input: math.tanh(input)
    if type == "elu":
        return lambda input: input if input > 0 else math.e ** input - 1

class Node:
    def __init__(self, num_next_nodes):
        self.num_next_nodes = num_next_nodes

class Dense:
    def __init__(self, n_neurons, activation_func, weight_init=1):
        self.n_neurons = n_neurons
        self.activation_func = activation_func
        self.weights = np.array([weight_init for _ in range(n_neurons)])

    def __call__(self, inputMat, inputWeights):
        output = sum([inputWeights[i] * inputMat[i] for i in range(self.n_neurons)])

