import collections
import random
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from actor_critic import ActorCritic, CartPoleEnv
from repnet.RepTree import TreeRL

num_networks_trained = 100

reward_threshold = 195


def train_repnet():
    cartpole = CartPoleEnv()

    seed = random.randrange(0, 100)
    cartpole.env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    min_episodes_criterion = 140
    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    num_actions = cartpole.env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    trunk = TreeRL(100, lambda x: 25, model)

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
    print(len(performance_list))
    return performance_list

def train_repnet():
    cartpole = CartPoleEnv()

    seed = random.randrange(0, 100)
    cartpole.env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    min_episodes_criterion = 140
    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    num_actions = cartpole.env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    trunk = TreeRL(100, lambda x: 25, model)

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
    print(len(performance_list))
    return performance_list

def train_rl():
    cartpole = CartPoleEnv()

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    min_episodes_criterion = 100
    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    num_actions = cartpole.env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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
        return train_repnet()
    return performance_list

def train_rl_hardstop(hardstop=300):
    cartpole = CartPoleEnv()

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    min_episodes_criterion = 100
    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    num_actions = cartpole.env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    performance_list = []

    iters = 0

    with tqdm.trange(max_episodes) as t:
        for i in t:
            if iters >= hardstop:
                return [hardstop]

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
        return train_repnet()
    return performance_list
