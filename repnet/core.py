import abc
import collections
import random
import statistics
import weakref

import numpy as np
import tensorflow as tf


class Branch(object):
    def __init__(self, max_kill_iters, threshold_func, performance=0, weights=None, generation=0):
        self.max_kill_iters = max_kill_iters
        self.reset_iters = 0

        self.threshold_func = threshold_func
        self.performance = performance
        self.weights = weights
        self.child_branches = []
        self.killed = False

        self.generation = generation

    def check_kill(self) -> None:
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

    def update(self, new_performance, new_weights) -> bool:
        self.check_kill()
        if self.killed:
            return True

        if new_performance - self.performance > self.threshold_func(self.performance):
            self.child_branches.append(Branch(self.max_kill_iters,
                                              self.threshold_func,
                                              performance=new_performance,
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
                       f"Performance: {self.performance}, Iterations since reset: {self.reset_iters}\n"
        for branch in self.child_branches:
            for i in range(self.generation):
                ret_str += "    "
            # '↳' is a UTF-8 character
            ret_str += "↳" + branch.__str__()

        return ret_str


class Tree:
    def __init__(self, max_kill_iters, threshold_func):
        self.main = Branch(max_kill_iters, threshold_func)

        self.best_weights = None
        self.best_performance = 0

        self.max_steps_per_episode = 1000

    def get_num_branch_ends(self):
        return self.main.get_branch_num_ends

    def get_branch_ends(self) -> np.array:
        return self.main.get_branch_ends

    def update_end(self, end, performance, weights) -> bool:
        if performance > self.best_performance:
            self.best_weights = weights
            self.best_performance = performance
        return end.update(performance, weights)


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
        performance_list = []
        episodes_reward: collections.deque = collections.deque(maxlen=self.min_episodes_criterion)

        for i in range(self.max_episodes):
            for end in self.tree.main.get_branch_ends():
                if end.killed:
                    continue

                if end.weights is not None:
                    self.model.set_weights(end.weights)

                initial_state = tf.constant(env_obj.env.reset(), dtype=tf.float32)
                episode_reward = int(env_obj.train_step(initial_state, self.model, self.optimizer, self.gamma, self.max_steps_per_episode))

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                performance_list.append(running_reward)

                end.weights = self.model.get_weights()

                if running_reward > self.running_reward_threshold:
                    return performance_list, end.weights

                self.tree.update_end(end, episode_reward, self.model.get_weights())
        if performance_list[-1] < self.running_reward_threshold:
            # TODO: fix the error that's occurring here,
            #  returning none instead of returning the performance list over time
            return None # self.train(env_obj)
        return performance_list, self.model.get_weights
