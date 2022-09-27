class Branch:
    def __init__(self, max_kill_iters, threshold_func=lambda P: 0, generation=1, model=None, performance=0):
        self.generation = generation

        self.model = model
        self.prev_model = None
        self.performance = performance
        self.prev_performance = 0
        self.reset_iters = 0

        self.killed = False
        self.max_kill_iters = max_kill_iters

        self.child_branches = []

        self.threshold_func = threshold_func

    def check_kill(self):
        if self.reset_iters > self.max_kill_iters:
            self.killed = True

    def update_branch(self, model, performance):
        if self.killed:
            return

        if performance - self.performance > self.threshold_func(self.performance):
            self.child_branches.append(Branch(self.max_kill_iters,
                                              generation=self.generation + 1,
                                              threshold_func=self.threshold_func,
                                              model=model,
                                              performance=performance
                                              ))
            self.reset_iters = 0


        # else:
        #     self.prev_model = model
        #     self.prev_performance = self.performance
        #
        #     self.model = model
        #     self.performance = performance


        self.reset_iters += 1

    def __str__(self):
        ret_str = f"Generation {self.generation}, Performance: {self.performance}, " \
                  f"Iterations since reset: {self.reset_iters}\n"
        for branch in self.child_branches:
            for i in range(self.generation):
                ret_str += "    "
            ret_str += "↳"+branch.__str__()

        return ret_str

