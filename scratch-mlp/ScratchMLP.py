import numpy as np
import math

def activation(type: str):
    if type == "Sigmoid":
        return lambda input: 1 / (1 + math.e ** (-input))
    if type == "ReLU":
        return lambda input: max(0, input)
    if type == "Leaky ReLU":
        return lambda input: input if input > 0 else input * 0.01


class DenseNode():
    def __init__(self, weight_init=1, activation_type="Sigmoid"):
        self.weight_out = weight_init

        self.bias = 1

        self.activation_func = activation(activation_type)

    def __call__(self, listInputs: list, listWeights: list):
        return self.activation_func(self.bias + sum(np.matmul(listInputs, listWeights)))


class Dense:
    def __init__(self, n_neurons, prev_layer=None, activation_type="Sigmoid"):
        self.neurons = [DenseNode(activation_type=activation_type) for _ in range(n_neurons)]

        self.prev_layer = prev_layer

    def __call__(self, inputs):
        if len(inputs) != len(self.neurons):
            raise Exception("Mismatched neurons/inputs!")

        if self.prev_layer is None:
            for i in range(len(self.neurons)):  # For input layer
                return [self.neurons[i]([inputs[i]], [1])]

        for i in range(len(self.neurons)):  # For other layers
            return [self.neurons[i]([inputs[i]], [])]