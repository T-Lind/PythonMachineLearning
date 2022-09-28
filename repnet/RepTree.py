from tensorflow.python.keras.models import clone_model
import gym
ID = 0


class Branch:
    def __init__(self, max_kill_iters, threshold_func=lambda P: 0, generation=1, model=None, performance=0, env_name=None):
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
            ret_list.append([values[i:i+2]])

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


def repnet(max_kill_iters, env_name, threshold_func):
    return Branch(max_kill_iters, threshold_func=threshold_func, env_name=env_name)
