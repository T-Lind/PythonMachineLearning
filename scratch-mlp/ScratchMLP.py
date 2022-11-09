import math


def activation(type: str):
    if type == "Sigmoid":
        return lambda input: 1 / (1 + math.e ** (-input))
    if type == "ReLU":
        return lambda input: max(0, input)
    if type == "Leaky ReLU":
        return lambda input: input if input > 0 else input * 0.01


class Node:
    def __init__(self, activation_type="Sigmoid"):
        self.weights_out = []

        self.activation_func = activation(activation_type)

    def call(self, listInputs: list):
        return self.activation_func(sum(listInputs))
