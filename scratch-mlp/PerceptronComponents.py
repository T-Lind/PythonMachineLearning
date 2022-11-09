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


class Layer:
    pass


class Input(Layer):
    def __init__(self, n_inputs, next_neurons):
        self.weights = [1] * n_inputs * next_neurons

    def __call__(self, input):
        return input


class Dense(Layer):
    def __init__(self, n_neurons, next_neurons, activation_func, weight_init=1):
        self.n_neurons = n_neurons
        self.activation_func = activation_func
        self.weights = np.array([weight_init for _ in range(n_neurons * next_neurons)])

    def __call__(self, input_mat, past_layer):
        input_weights = past_layer.weights
        output_values = [0] * self.n_neurons

        assert len(input_mat) * self.n_neurons == len(input_weights)

        for i in range(self.n_neurons):
            j_vals = [j for j in range(len(input_mat))]
            k_vals = [k for k in range(i, len(input_weights), self.n_neurons)]

            neuron_summation = 0
            for q in range(len(j_vals)):
                input_idx, weight_idx = j_vals[q], k_vals[q]
                neuron_summation += input_mat[input_idx] * input_weights[weight_idx]
            output_values[i] = self.activation_func(neuron_summation)

        return output_values




layer_1 = Input(3, 3)
layer_2 = Dense(3, 2, activation_func=activation("relu"))
layer_3 = Dense(2, 2, activation_func=activation("relu"))

out_1 = layer_2([1, 2, 3], layer_1)
print(out_1)
out_2 = layer_3(out_1, layer_2)
print(out_2)

