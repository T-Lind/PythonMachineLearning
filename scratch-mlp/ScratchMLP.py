from mlp_lib.PerceptronComponents import Network, Dense, Input
from mlp_lib.Functions import activation

# Example network
network = Network([
    Input(3, 3),
    Dense(3, 5, activation_func=activation("leaky_relu")),
    Dense(5, 128, activation_func=activation("sigmoid")),
    Dense(128, 2, activation_func=activation("sigmoid")),
    Dense(2, 1, activation_func=activation("relu")),
])
print(network.predict([3, 2, 1]))
