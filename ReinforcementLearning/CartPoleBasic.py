import gym
import keyboard
import numpy as np
import logging

# Load the CartPole game from OpenAIGym
env = gym.make('CartPole-v1', render_mode='human')

obs = env.reset()


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1



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
            exit()
    totals.append(episode_rewards)


print(f"Max steps: {np.max(totals)}, Min steps: {np.min(totals)}, Std: {np.std(totals)}, Average: {np.mean(totals)}")