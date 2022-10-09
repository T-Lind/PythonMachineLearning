from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


environment = suite_gym.load('CartPole-v0')
print('action_spec:', environment.action_spec())
print('time_step_spec.observation:', environment.time_step_spec().observation)
print('time_step_spec.step_type:', environment.time_step_spec().step_type)
print('time_step_spec.discount:', environment.time_step_spec().discount)
print('time_step_spec.reward:', environment.time_step_spec().reward)

# action = np.array(1, dtype=np.int32)
# time_step = environment.reset()
# print(time_step)
# while not time_step.is_last():
#   time_step = environment.step(action)
#   print(time_step)
#
#

env = suite_gym.load('CartPole-v0')
tf_env = tf_py_environment.TFPyEnvironment(env)

time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 5

for _ in range(num_episodes):
  episode_reward = 0
  episode_steps = 0
  while not time_step.is_last():
    # action = tf.random.uniform([1], 0, 2, dtype=tf.int32)
    action = np.array(1, dtype=np.int32)
    time_step = tf_env.step(action)
    episode_steps += 1
    episode_reward += time_step.reward.numpy()
  rewards.append(episode_reward)
  steps.append(episode_steps)
  time_step = tf_env.reset()

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)
