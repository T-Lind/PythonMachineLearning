import collections
import random
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from repnet.core import Tree
from repnet.ActorCritic import ActorCritic
from repnet.examples.Cartpole import CartPoleEnv

def train_repnet(threshold_func=lambda x: 100, kill_time=100, gamma=0.99, max_episodes=10000, learning_rate=0.01,
                 reward_threshold=195, seed=None):
    """
    Trains a basic REPNET
    :param seed: the randomizng seed to train on
    :param threshold_func: the input function to deterimine when to create a new branch
    :param kill_time: the maximum number of episodes from the last time of reproduction
    that a new branch can exist for until it dies
    :param gamma: the discount factor for future rewards
    :param max_episodes: the maximum amount of episodes that can be trained until the training session exits (hard stop)
    :param learning_rate: The learning rate of the Adam optimizer
    :param reward_threshold: The minimum running reward needed to consider the model trained
    :return: A list of the running reward every episode
    """
    cartpole = CartPoleEnv()

    if seed is None:
        seed = random.randrange(0, 100)
        # cartpole.env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    min_episodes_criterion = 140
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    running_reward = 0

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    num_actions = cartpole.env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    trunk = Tree(kill_time, threshold_func, model)

    for end in trunk.get_branch_ends():
        end.weights = model.get_weights()

    performance_list = []

    for i in range(max_episodes):
        for end in trunk.get_branch_ends():
            if end.killed:
                continue

            if end.weights is not None:
                model.set_weights(end.weights)

            initial_state = tf.constant(cartpole.env.reset(), dtype=tf.float32)
            episode_reward = int(cartpole.train_step(
                initial_state, model, optimizer, gamma, max_steps_per_episode))

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            performance_list.append(running_reward)

            end.weights = model.get_weights()

            if running_reward > reward_threshold:
                return performance_list

            trunk.update_end(end, episode_reward, model.get_weights())
    if performance_list[-1] < reward_threshold:
        return train_repnet()
    return performance_list


def train_distributed_repnet(threshold_func=lambda x: 100, kill_time=100, gamma=0.99, max_episodes=10000,
                             learning_rate=0.01, reward_threshold=195, seed=None):
    """
    Trains REPNET based on the multiprocessing distribution principle
    :param seed: the randomizing seed to train on
    :param threshold_func: the input function to deterimine when to create a new branch
    :param kill_time: the maximum number of episodes from the last time of reproduction
    that a new branch can exist for until it dies
    :param gamma: the discount factor for future rewards
    :param max_episodes: the maximum amount of episodes that can be trained until the training session exits (hard stop)
    :param learning_rate: The learning rate of the Adam optimizer
    :param reward_threshold: The minimum running reward needed to consider the model trained
    :return: A list of the running reward every episode
    """
    cartpole = CartPoleEnv()

    if seed is None:
        seed = random.randrange(0, 100)
    # cartpole.env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    min_episodes_criterion = 140
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    running_reward = 0

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    num_actions = cartpole.env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    trunk = Tree(kill_time, threshold_func, model)

    for end in trunk.get_branch_ends():
        end.weights = model.get_weights()

    performance_list = []

    iters = 0
    for i in range(max_episodes):
        performance_list.append(running_reward)
        for end in trunk.get_branch_ends():
            if end.killed:
                continue

            if end.weights is not None:
                model.set_weights(end.weights)
            iters += 1

            initial_state = tf.constant(cartpole.env.reset(), dtype=tf.float32)
            episode_reward = int(cartpole.train_step(
                initial_state, model, optimizer, gamma, max_steps_per_episode))

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            end.weights = model.get_weights()

            if running_reward > reward_threshold:
                return performance_list

            trunk.update_end(end, episode_reward, model.get_weights())
    if performance_list[-1] < reward_threshold:
        return train_distributed_repnet()
    print(len(performance_list))
    return performance_list


def train_rl(gamma=0.99, max_episodes=10000, learning_rate=0.01, reward_threshold=195, seed=None):
    """
    Trains a normal actor-critic network
    :param seed: The randomizing seed to train on
    :param gamma: the discount factor for future rewards
    :param max_episodes: the maximum amount of episodes that can be trained until the training session exits (hard stop)
    :param learning_rate: The learning rate of the Adam optimizer
    :param reward_threshold: The minimum running reward needed to consider the model trained
    :return: A list of the running reward every episode
    """
    cartpole = CartPoleEnv()

    if seed is None:
        seed = random.randrange(0, 100)
    # cartpole.env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    min_episodes_criterion = 100
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    num_actions = cartpole.env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    performance_list = []

    iters = 0

    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(cartpole.env.reset(), dtype=tf.float32)
            episode_reward = int(cartpole.train_step(
                initial_state, model, optimizer, gamma, max_steps_per_episode))

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            iters += 1

            performance_list.append(running_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass  # print(f'Episode {i}: average reward: {avg_reward}')

            if running_reward > reward_threshold and i >= min_episodes_criterion:
                break
    if performance_list[-1] < reward_threshold:
        return train_rl()
    return performance_list