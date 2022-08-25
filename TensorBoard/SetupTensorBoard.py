import os

import numpy as np
from tensorflow import keras
import tensorflow as tf

root_logdir = "C:\\Users\\zenith\\PycharmProjects\\MachineLearning\\TensorBoard\\my_logs"



def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

np.random.seed(42)
tf.random.set_seed(42)
