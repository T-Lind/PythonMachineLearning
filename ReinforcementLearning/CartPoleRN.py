import gym
import keyboard
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the CartPole game from OpenAIGym
env = gym.make('CartPole-v1')
obs = env.reset()

# Create RN Learning model
n_inputs = env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation='elu', input_shape=[n_inputs]),
    keras.layers.Dense(1, activation='sigmoid'),
])




totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
        if keyboard.is_pressed('q'):
            break
    if keyboard.is_pressed('q'):
        break
    totals.append(episode_rewards)

print(f"Max steps: {np.max(totals)}, Min steps: {np.min(totals)}, Std: {np.std(totals)}, Average: {np.mean(totals)}")