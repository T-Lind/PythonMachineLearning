import gym
from gym import spaces
import pygame
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # Size of square grid

