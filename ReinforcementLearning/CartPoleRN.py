import time

import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import sys
import warnings


if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Load the CartPole game from OpenAIGym
env = gym.make('CartPole-v1')
obs = env.reset()

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

n_inputs = env.observation_space.shape[0]

n_environments = 50


envs = [gym.make("CartPole-v1") for _ in range(n_environments)]
for index, env in enumerate(envs):
    env.seed(index)
np.random.seed(42)
observations = [env.reset() for env in envs]
optimizer = keras.optimizers.RMSprop()
loss_fn = keras.losses.binary_crossentropy

training_times = []


def train_reinforcement_network(n_layers, n_neurons, thresh=0.01):
    network_layers = [keras.layers.Dense(5, activation="elu", input_shape=[n_inputs])]
    for i in range(n_layers):
        network_layers.append(keras.layers.Dense(n_neurons, activation="relu", kernel_initializer="he_normal"))
        network_layers.append(keras.layers.BatchNormalization())
    network_layers.append(keras.layers.Dense(1, activation="softmax"))
    model = keras.models.Sequential(network_layers)

    start_time = time.time()
    iteration = 0
    loss = 1
    while loss > thresh:
        # if angle < 0, we want proba(left) = 1., or else proba(left) = 0.
        target_probas = np.array([([1.] if obs[2] < 0 else [0.])
                                  for obs in observations])
        with tf.GradientTape() as tape:
            left_probas = model(np.array(observations))
            loss = tf.reduce_mean(loss_fn(target_probas, left_probas))
        print("\rIteration: {}, Loss: {:.3f}".format(iteration, loss.numpy()), end="")
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        actions = (np.random.rand(n_environments, 1) > left_probas.numpy()).astype(np.int32)
        for env_index, env in enumerate(envs):
            obs, reward, done, info = env.step(actions[env_index][0])
            observations[env_index] = obs if not done else env.reset()

        if time.time()-start_time > 60*4:
            break

        iteration += 1

    if time.time()-start_time <= 60*4:
        train_time = time.time() - start_time
        training_times.append(train_time)
    else:
        training_times.append(-1)

    for env in envs:
        env.close()

    model.save('CartPole')


layers = 4
neurons = 10


train_reinforcement_network(layers, neurons, thresh=0.003)

print("\n", training_times)
