import ftc_field
import gym
from tensorflow import keras

env = gym.make('ftc_field/ftc_field-v0', render_mode='human')

obs = env.reset()

n_layers = 6
n_neurons = 128
n_inputs =

network_layers = [keras.layers.Dense(5, activation="elu", input_shape=[n_inputs])]
for i in range(n_layers):
    network_layers.append(keras.layers.Dense(n_neurons, activation="relu", kernel_initializer="he_normal"))
    network_layers.append(keras.layers.BatchNormalization())
network_layers.append(keras.layers.Dense(21, activation="softmax"))
model = keras.models.Sequential(network_layers)

terminated = False

while not terminated:
    action = 3
    if type(obs) == dict:
        print("Agent red pos: ", obs["agent_red"], "Agent red carrying", obs["carrying"][0])
    obs, reward, terminated, info = env.step(action)

env.close()
