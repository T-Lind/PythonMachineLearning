from tensorflow.python.keras.models import clone_model
import gym
import numpy as np

ID = 0


class Branch:

    @DeprecationWarning
    def __init__(self, max_kill_iters, threshold_func=lambda P: 0, generation=1, model=None, performance=0,
                 env_name=None):
        global ID
        self.generation = generation

        self.id = ID
        ID += 1

        self.model = model
        self.prev_model = None
        self.performance = performance
        self.prev_performance = 0
        self.reset_iters = 0

        self.killed = False
        self.max_kill_iters = max_kill_iters

        self.child_branches = []

        self.threshold_func = threshold_func

        self.env_name = env_name
        if self.env_name is not None:
            self.env = gym.make(self.env_name)
            self.env.reset()

    def check_kill(self):
        if self.reset_iters > self.max_kill_iters:
            self.killed = True

    def label_models(self):
        id_performances = [self.performance, self.model, self.env]
        for branch in self.child_branches:
            id_performances += branch.label_models()
        return id_performances

    def best_models(self, top_models=10):
        values = self.label_models()
        ret_list = []
        for i in range(0, len(values), 3):
            ret_list.append([values[i:i + 2]])

        if len(ret_list) >= top_models:
            return sorted(ret_list, reverse=True)[:top_models]
        return sorted(ret_list, reverse=True)

    def update_branch(self, performance):
        self.check_kill()

        if self.killed:
            return True

        if performance - self.performance > self.threshold_func(self.performance):
            self.child_branches.append(Branch(self.max_kill_iters,
                                              generation=self.generation + 1,
                                              threshold_func=self.threshold_func,
                                              model=clone_model(self.model),
                                              performance=performance,
                                              env_name=self.env_name
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
        ret_str += f"Generation {self.generation}, ID: {self.id}," \
                   f"Performance: {self.performance}, Iterations since reset: {self.reset_iters}\n"
        for branch in self.child_branches:
            for i in range(self.generation):
                ret_str += "    "
            ret_str += "↳" + branch.__str__()

        return ret_str


class MinBranch:
    def __init__(self, max_kill_iters, threshold_func, performance=0, weights=None, generation=0):
        self.max_kill_iters = max_kill_iters
        self.reset_iters = 0

        self.threshold_func = threshold_func
        self.performance = performance
        self.weights = weights
        self.child_branches = []
        self.killed = False

        self.generation = generation

    def check_kill(self):
        if self.reset_iters > self.max_kill_iters:
            self.killed = True

    def get_branch_num_ends(self):
        if len(self.child_branches) == 0:
            return 1

        branch_end_sum = 0
        for child in self.child_branches:
            branch_end_sum += child.get_branch_num_ends()
        return branch_end_sum

    def get_branch_ends(self):
        if len(self.child_branches) == 0:
            return self

        branch_ends = []
        for child in self.child_branches:
            branch_ends.append(child.get_branch_ends())
        return np.array(branch_ends).flatten()

    def update(self, new_performance, new_weights):
        self.check_kill()
        if self.killed:
            return True

        if new_performance - self.performance > self.threshold_func(self.performance):
            self.child_branches.append(MinBranch(self.max_kill_iters,
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
        ret_str += f"Generation {self.generation}," \
                   f"Performance: {self.performance}, Iterations since reset: {self.reset_iters}\n"
        for branch in self.child_branches:
            for i in range(self.generation):
                ret_str += "    "
            ret_str += "↳" + branch.__str__()

        return ret_str


class Tree:
    def __init__(self, max_kill_iters, threshold_func, model_baseline, train_x, train_Y, test_x, test_Y):
        self.main = MinBranch(max_kill_iters, threshold_func)
        self.model_baseline = model_baseline
        self.train_x = train_x
        self.train_Y = train_Y
        self.test_x = test_x
        self.test_Y = test_Y

    def get_num_branch_ends(self):
        return self.main.get_branch_num_ends()

    def update_branch_ends(self):
        best_acc = 0
r
        ends = self.main.get_branch_ends()

        if type(ends) == MinBranch:
            self.model_baseline.fit(self.train_x, self.train_Y)
            accuracy = self.model_baseline.evaluate(self.test_x, self.test_Y, verbose=2)[1]
            if accuracy > best_acc:
                best_acc = accuracy
            ends.update(accuracy, self.model_baseline.get_weights())
            return best_acc

        for i in range(len(ends)-1, -1, -1):
            end = ends[i]
            if end.killed:
                continue

            print(end)
            if end.weights is not None:
                self.model_baseline.set_weights(end.weights)
            self.model_baseline.fit(self.train_x, self.train_Y)

            accuracy = self.model_baseline.evaluate(self.test_x, self.test_Y, verbose=2)[1]
            if accuracy > best_acc:
                best_acc = accuracy
            end.update(accuracy, self.model_baseline.get_weights())

        return best_acc

    def __str__(self):
        return str(self.main)


def repnet_rl(max_kill_iters, env_name, threshold_func):
    return Branch(max_kill_iters, threshold_func=threshold_func, env_name=env_name)


def repnet(max_kill_iters, threshold_func):
    return Branch(max_kill_iters, threshold_func=threshold_func)
