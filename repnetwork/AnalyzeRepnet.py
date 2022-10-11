import collections
import random
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from actor_critic import train_step, ActorCritic, create_env
from repnet.RepTree import TreeRL
import matplotlib.pyplot as plt

num_networks_trained = 5


def merge_lists(list1, list2):
    merged_list = [0] * max(len(list1), len(list2))

    for i in range(len(list1)):
        merged_list[i] += list1[i]
    for i in range(len(list2)):
        merged_list[i] += list2[i]
    return merged_list


def train_repnet():
    env = create_env()

    seed = random.randrange(0, 100)
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    min_episodes_criterion = 140
    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    reward_threshold = 195
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    num_actions = env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    trunk = TreeRL(100, lambda x: 50, model)

    for end in trunk.get_branch_ends():
        end.weights = model.get_weights()

    performance_list = []

    iters = 0
    for i in range(max_episodes):
        # print(trunk.get_branch_ends())
        for end in trunk.get_branch_ends():
            if end.killed:
                continue

            if end.weights is not None:
                model.set_weights(end.weights)
            iters += 1

            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            episode_reward = int(train_step(
                initial_state, model, optimizer, gamma, max_steps_per_episode))

            # print(episode_reward)

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            performance_list.append(running_reward)

            end.weights = model.get_weights()

            if running_reward > reward_threshold:
                return performance_list

            trunk.update_end(end, episode_reward, model.get_weights())
    return performance_list


result = train_repnet()
print(result)
print(len(result))
