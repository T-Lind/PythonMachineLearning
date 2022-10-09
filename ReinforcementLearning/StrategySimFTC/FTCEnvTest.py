import random

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

inputs = keras.Input(shape=(111, 1))
dense1 = keras.layers.Dense(256, activation="relu")(inputs)
dense2 = keras.layers.Dense(256, activation="relu")(dense1)
dense3 = keras.layers.Dense(512, activation="relu")(dense2)
dense4 = keras.layers.Dense(256, activation="relu")(dense3)
output = keras.layers.Dense(22, activation="softmax")(dense4)
model = keras.Model(inputs=inputs, outputs=output, name="FTCModel")

print(model.summary())

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Nadam(),
              metrics=['accuracy'])

terminated = False

single_feature_normalizer = tf.keras.layers.Normalization(axis=None)

while not terminated:
    action_possibilities = np.array([], dtype=np.int32)
    for single_observation in obs.values():
        action_possibilities = np.concatenate((np.array(single_observation).flatten(), action_possibilities))

    sparse_action_matrix = model(tf.expand_dims(action_possibilities, axis=0))
    greatest_prob = 0
    print(sparse_action_matrix[0][110])
    # for i in range(22):
    #     if sparse_action_matrix[0][110][i] > greatest_prob:
    #         greatest_prob = sparse_action_matrix[0][110][i]
    #         action = i

    obs, reward, terminated, info = env.step(np.array(sparse_action_matrix[0][110], dtype=np.int32))

env.close()
