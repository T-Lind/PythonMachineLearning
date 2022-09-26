import gym
from gym import spaces
import pygame
import numpy as np


class PoleState:
    def __init__(self):
        self.capped = False
        self.pole_owner = None
        self.red_scored = 0
        self.blue_scored = 0


# Only two teams for now
class FieldEnvFTC(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        # Number of tiles on a side
        self.size = 6
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "agent_red": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "agent_blue": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "pole_states": spaces.Box(0, self.size - 1, shape=(5, 5), dtype=PoleState),
                "corner_states": spaces.Box(0, self.size - 1, shape=(4,), dtype=bool),
                "end_game": spaces.Box(0, 1, shape=(1,), dtype=bool)
            }
        )

        # right/up/left/down & deposit/cap NE/NW/SW/SE for red/blue agent
        # 0-7 is move for red/blue, 8-23 us deposit/cap for red/blue, 24-25 is intake red/blue, 26-29 is for the corners (NE/NW/SW/SE)

        self.action_space = spaces.Discrete(30)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent_red": self._agent_red_location, "agent_blue": self._agent_blue_location,
                "pole_states": self._pole_states, "corner_states": self._corner_states, "end_game": self._end_game}

    def _get_info(self):
        # Get the current scores
        return {"red_score": self._red_score, "blue_score": self._blue_score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Times the matches
        self._step_count = 0
        self._end_game = False

        self._agent_red_carrying = False
        self._agent_blue_carrying = False

        self._pole_states = [[PoleState()] * 5] * 5
        self._corner_states = [False] * 4

        self._agent_red_location = np.array(0, 1)
        self._agent_blue_location = np.array(5, 1)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # If we want to move agent red
        if action < 4:
            direction = self._action_to_direction[action]
            self._agent_red_location = np.clip(self._agent_red_location + direction, 0, self.size - 1)

        # If we want to move agent blue
        elif action < 8:
            direction = self._action_to_direction[action - 4]
            self._agent_blue_location = np.clip(self._agent_blue_location + direction, 0, self.size - 1)

        # If we want red to deposit
        elif action < 12:
            redef_action = 12 - action

            if self._agent_red_location[0] == 0 and self._agent_red_location[1] == 0:
                self._corner_states[1] = True

            elif self._agent_red_location[0] == 5 and self._agent_red_location[1] == 5:
                self._corner_states[3] = True

            elif redef_action == 0 and self._agent_red_carrying:
                deposit_r = self._agent_red_location[0] - 1
                deposit_c = self._agent_red_location[1]

                # Must be in bounds to be able to score on a pole
                if 0 <= deposit_r < 6 and 0 <= deposit_c < 6:
                    self._pole_states[deposit_r][deposit_c].red_scored += 1
