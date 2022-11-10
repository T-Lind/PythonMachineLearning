import math
import numpy as np
from mlp_lib.Functions import activation, loss, optimizer


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


class Network:
    def __init__(self, layer_array):
        self.layers = layer_array
    def predict(self, input_data):
        result = input_data
        for i in range(len(self.layers)):
            if i > 0:
                result = self.layers[i](result, self.layers[i-1])
            else:
                result = self.layers[i](result)
        return result


# Example network
network = Network([
    Input(3, 3),
    Dense(3, 5, activation_func=activation("leaky_relu")),
    Dense(5, 128, activation_func=activation("sigmoid")),
    Dense(128, 2, activation_func=activation("sigmoid")),
    Dense(2, 1, activation_func=activation("relu")),
])
print(network.predict([3, 2, 1]))
