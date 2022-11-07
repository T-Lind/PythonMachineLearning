import collections
import random
import statistics
from typing import Callable

import numpy as np
import tensorflow as tf

from repnet.ActorCritic import ActorCritic


class Branch:
    def __init__(self, max_kill_iters, threshold_func, running_reward=0, weights=None, generation=0):
        """
        Create a branch object
        :param max_kill_iters: the kill time hyperparameter
        :param threshold_func: a threshold function to determine creation of new branches
        :param running_reward: the performance of the branch
        :param weights: the current weigths the branch containts
        :param generation: the generation the branch is. The first generation is 0 and increments by one.
        """
        self.max_kill_iters = max_kill_iters
        self.reset_iters = 0

        self.threshold_func = threshold_func
        self.running_reward = running_reward
        self.weights = weights
        self.child_branches = []
        self.killed = False

        self.generation = generation

    def check_kill(self) -> None:
        """
        Determine
        :return:
        """
        if self.reset_iters > self.max_kill_iters:
            self.killed = True

        for child in self.child_branches:
            if len(child.child_branches) == 0:
                del child

    # @DeprecationWarning
    def get_branch_num_ends(self) -> int:
        if len(self.child_branches) == 0:
            return 1

        branch_end_sum = 0
        for child in self.child_branches:
            branch_end_sum += child.get_branch_num_ends()
        return branch_end_sum

    # @DeprecationWarning
    def get_branch_ends(self) -> np.array:
        if len(self.child_branches) == 0:
            return [self]

        branch_ends = []
        for child in self.child_branches:
            branch_ends.append(child.get_branch_ends())
        return np.array(branch_ends).flatten()

    def update(self, new_running_reward, new_weights) -> bool:
        self.check_kill()
        if self.killed:
            return True

        if new_running_reward - self.running_reward > self.threshold_func(self.running_reward):
            self.child_branches.append(Branch(self.max_kill_iters,
                                              self.threshold_func,
                                              running_reward=new_running_reward,
                                              weights=new_weights,
                                              generation=self.generation + 1
                                              ))
            self.reset_iters = 0
        self.reset_iters += 1
        return False

    def __str__(self):
        ret_str = ""

        # Check and see if main branch, if so specify
        if self.generation == 1:
            ret_str += "trunk: "
        else:
            ret_str += "branch: "

        if not self.killed:
            ret_str += f"Generation {self.generation}," \
                       f"Running reward: {self.running_reward}, Iterations since reset: {self.reset_iters}\n"
        for branch in self.child_branches:
            for i in range(self.generation):
                ret_str += "    "
            # '↳' is a UTF-8 character
            ret_str += "↳" + branch.__str__()

        return ret_str


class Tree:
    def __init__(self, max_kill_iters, threshold_func):
        """
        Initialize the tree object
        :param max_kill_iters: the kill time hyperparameter
        :param threshold_func: a reference to a threshold function
        """
        self.main = Branch(max_kill_iters, threshold_func)

        self.best_weights = None
        self.best_running_reward = 0

        self.max_steps_per_episode = 1000

    def get_num_branch_ends(self) -> Callable[[], int]:
        """
        Get the number of total child branches that exist from this branch
        :return: the # of branches from this branch
        """
        return self.main.get_branch_num_ends

    def get_branch_ends(self) -> np.array:
        """
        Get a list of references to each branch end in a list
        :return: a list of branch references
        """
        return self.main.get_branch_ends

    def update_end(self, end, running_reward, weights) -> bool:
        """
        Update a certain branch end as provided through an argument
        :param end: the reference to the branch object
        :param running_reward: the performance of the weights stored in the branch
        :param weights: a reference of the weights
        :return: a boolean of whether the branch was updated or not
        """
        if running_reward > self.best_running_reward:
            self.best_weights = weights
            self.best_running_reward = running_reward
        return end.update(running_reward, weights)


@DeprecationWarning
class Repnet:

    def __init__(self, tree=None, gamma=0.99, max_episodes=10000,
                 max_steps_per_episode=1000, model=None, optimizer=None,
                 seed=None, min_episodes_criterion=140, reward_threshold=195
                 ):
        if seed is None:
            seed = random.randrange(0, 1000)
            tf.random.set_seed(seed)
            np.random.seed(seed)

        self.tree = tree
        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode
        self.min_episodes_criterion = min_episodes_criterion
        self.max_episodes = max_episodes
        self.running_reward_threshold = reward_threshold

        self.model = model
        self.optimizer = optimizer

        self.tree.main.weights = model.get_weights()

    def train(self, env_obj):
        """
        Train the repnet object on the environment object provided. see repnet.examples.Cartpole for an example
        :param env_obj: the environment object provided
        :return: a list of running reward over time
        """
        running_reward_list = []
        episodes_reward: collections.deque = collections.deque(maxlen=self.min_episodes_criterion)

        for i in range(self.max_episodes):
            for end in self.tree.main.get_branch_ends():
                if end.killed:
                    continue

                if end.weights is not None:
                    self.model.set_weights(end.weights)

                initial_state = tf.constant(env_obj.env.reset(), dtype=tf.float32)
                episode_reward = int(env_obj.train_step(initial_state, self.model, self.optimizer,
                                                        self.gamma, self.max_steps_per_episode))

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                running_reward_list.append(running_reward)

                end.weights = self.model.get_weights()

                if running_reward > self.running_reward_threshold:
                    return running_reward_list, end.weights

                self.tree.update_end(end, episode_reward, self.model.get_weights())
        if running_reward_list[-1] < self.running_reward_threshold:
            # TODO: fix the error that's occurring here,
            #  returning none instead of returning the performance list over time
            return self.train(env_obj)
        return running_reward_list, self.model.get_weights


def train(env_obj=None, max_episodes=10000, tree=None, model=None, optimizer=None, gamma=0.99,
          max_steps_per_episode=1000,
          running_reward_threshold=None, min_episodes_criterion=140) -> (list, np.ndarray):
    """
    Train the repnet object on the environment object provided. see repnet.examples.Cartpole for an example
    :param env_obj: the environment object provided that MUST have a train_step function and reference to gym env
    :param max_episodes: the maximum episodes of training before it exits
    :param tree: the Tree object to grow
    :param model: a reference to a basic model that can be trained
    :param optimizer: the optimization method to use
    :param gamma: the discount factor for future rewards
    :param max_steps_per_episode: the maximum amount of steps allowed for one training episode
    :param running_reward_threshold: when the function should quit
    :param min_episodes_criterion: the minimum episodes required
    :return: a list of running reward over time
    """

    running_reward_list = []
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    running_reward = 0

    tree.main.weights = model.get_weights()

    for i in range(max_episodes):
        if tree.main.get_branch_num_ends() == 0:
            return train(env_obj=env_obj, max_episodes=max_episodes,
                         tree=Tree(tree.main.max_kill_iters, tree.main.threshold_func), model=ActorCritic(2, 128),
                         optimizer=optimizer,
                         gamma=gamma, max_steps_per_episode=max_steps_per_episode,
                         running_reward_threshold=running_reward_threshold)

        for end in tree.main.get_branch_ends():
            if end.killed:
                continue

            if end.weights is not None:
                model.set_weights(end.weights)

            initial_state = tf.constant(env_obj.env.reset(), dtype=tf.float32)
            episode_reward = int(env_obj.train_step(initial_state, model, optimizer,
                                                    gamma, max_steps_per_episode))

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            running_reward_list.append(running_reward)

            end.weights = model.get_weights()

            if running_reward > running_reward_threshold:
                return running_reward_list, end.weights

            tree.update_end(end, episode_reward, model.get_weights())
    if running_reward_list[-1] < running_reward_threshold:
        # TODO: fix the error that's occurring here,
        #  returning a length of 1 instead of returning the performance list over time
        return train(env_obj=env_obj, max_episodes=max_episodes,
                     tree=Tree(tree.main.max_kill_iters, tree.main.threshold_func), model=ActorCritic(2, 128),
                     optimizer=optimizer,
                     gamma=gamma, max_steps_per_episode=max_steps_per_episode,
                     running_reward_threshold=running_reward_threshold)
    return running_reward_list, model.get_weights
