from repnet import repnet

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
env_load = gym.make('CartPole-v1')
observations = env_load.reset()

n_inputs = env_load.observation_space.shape[0]

main = repnet(50, "CartPole-v1", lambda P: -0.05 * P + 0.05)

def train_repnet(n_layers, n_neurons, thresh=0.01):
    network_layers = [keras.layers.Dense(5, activation="elu", input_shape=[n_inputs])]
    for i in range(n_layers):
        network_layers.append(keras.layers.Dense(n_neurons, activation="relu", kernel_initializer="he_normal"))
        network_layers.append(keras.layers.BatchNormalization())
    network_layers.append(keras.layers.Dense(1, activation="softmax"))

    # Model to use when training
    model = keras.models.Sequential(network_layers)

    main.model = model

    # Time out at 1000 runs
    for _ in range(1000):
        # Update performance information
        models_info = main.best_models(top_models=100)

        best_performance, best_model, env = models_info[0], models_info[0], 0
        print(best_performance, best_model, env)

        if best_performance < thresh:
            return best_performance, best_model

        target_probas = np.array([([1.] if obs[2] < 0 else [0.])
                                  for obs in observations])

        for model_info in models_info:
            indiv_model = model_info[1]

            optimizer = keras.optimizers.RMSprop()
            loss_fn = keras.losses.binary_crossentropy

            with tf.GradientTape() as tape:
                left_probas = indiv_model(np.array(observations))
                loss = tf.reduce_mean(loss_fn(target_probas, left_probas))
            main.update_branch(1-loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            actions = (np.random.rand(1, 1) > left_probas.numpy()).astype(np.int32)

            env.step(actions[0][0])



    return None

print(train_repnet(5, 50))
