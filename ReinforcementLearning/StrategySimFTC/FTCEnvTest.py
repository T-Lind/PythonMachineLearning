import np as np
import numpy as np
import tensorflow as tf
import ftc_field
import gym
from tensorflow import keras

env = gym.make('ftc_field/ftc_field-v0', render_mode='human')
n_inputs = env.action_space.shape
print(n_inputs)

obs = env.reset()

n_layers = 6
n_neurons = 128
n_inputs = 111

model = keras.Sequential(
    [
        keras.layers.Dense(111, activation="relu", name="layer1"),
        keras.layers.Dense(100, activation="relu", name="layer2"),
        keras.layers.Dense(22, name="layer3"),
    ]
)
optimizer = keras.optimizers.RMSprop()
loss_fn = keras.losses.binary_crossentropy

model.compile(optimizer, loss_fn)

terminated = False

single_feature_normalizer = tf.keras.layers.Normalization(axis=None)

while not terminated:
    action_possibilities = np.array([])
    for single_observation in obs.values():
        action_possibilities = np.concatenate((np.array(single_observation).flatten(), action_possibilities))
    action = model(single_feature_normalizer.adapt(action_possibilities))
    obs, reward, terminated, info = env.step(i for i in range(len(action)) if i == 1)

env.close()
