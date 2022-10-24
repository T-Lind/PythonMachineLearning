import numpy as np


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
            return [self]

        branch_ends = []
        for child in self.child_branches:
            branch_ends.append(child.get_branch_ends())
        return np.array(branch_ends).flatten()

    def update(self, new_performance, new_weights):
        self.check_kill()
        if self.killed:
            # del self
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

        if not self.killed:
            ret_str += f"Generation {self.generation}," \
                       f"Performance: {self.performance}, Iterations since reset: {self.reset_iters}\n"
        for branch in self.child_branches:
            for i in range(self.generation):
                ret_str += "    "
            ret_str += "↳" + branch.__str__()

        return ret_str

class TreeRL:
    def __init__(self, max_kill_iters, threshold_func, model_baseline):
        self.main = MinBranch(max_kill_iters, threshold_func)
        self.model_baseline = model_baseline

        self.best_weights = None
        self.best_performance = 0

    def get_num_branch_ends(self):
        return self.main.get_branch_num_ends()

    def get_branch_ends(self):
        return self.main.get_branch_ends()

    def update_end(self, end, performance, weights):
        if performance > self.best_performance:
            self.best_weights = weights
            self.best_performance = performance
        return end.update(performance, weights)