import gym
from gym import spaces
import pygame
import numpy as np


class PoleState:
    def __init__(self):
        # self.capped = False
        self.pole_owner = None
        self.red_scored = 0
        self.blue_scored = 0
        self.pole_type = None

    def obs(self):
        states = []
        if self.pole_owner == "red":
            states.append(1)
        elif self.pole_owner == "blue":
            states.append(0)

        states.append(self.red_scored)
        states.append(self.blue_scored)

        return states


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
        # 0-7 is move for red/blue, 8-15 us deposit for red/blue, 16-17 is intake red/blue, 18-21 is for the corners (NE/NW/SW/SE)

        self.action_space = spaces.Discrete(30)

        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        states = np.array([])
        for pole_row in self._pole_states:
            states += np.array([pole.obs() for pole in pole_row])
        num = 1
        for dim in states.shape:
            num *= dim
        states = states.reshape(1, num)

        return self._agent_red_location+self._agent_blue_location+states+np.array(self._corner_states)+np.array([int(self._agent_red_carrying)])+np.array([int(self._agent_blue_carrying)])
        # return {"agent_red": self._agent_red_location, "agent_blue": self._agent_blue_location,
        #         "pole_states": self._pole_states, "corner_states": self._corner_states, "end_game": self._end_game,
        #         "carrying": (self._agent_red_carrying, self._agent_blue_carrying)
        #         }

    def _get_info(self):
        # Get the current scores
        red_score, blue_score = self._calculate_score()
        return {"red_score": red_score, "blue_score": blue_score}

    def _score(self, r, c, team):
        # Must be in bounds to be able to score on a pole
        self._pole_states[r][c].pole_owner = team
        if 0 <= r < 6 and 0 <= c < 6:
            if team == "red":
                self._pole_states[r][c].red_scored += 1
            elif team == "blue":
                self._pole_states[r][c].blue_scored += 1

    def _calculate_score(self):
        red_score = 0
        blue_score = 0

        # Count terminal points
        if self._corner_states[1]:
            red_score += 1
        if self._corner_states[3]:
            red_score += 1

        # Count terminal points
        if self._corner_states[0]:
            blue_score += 1
        if self._corner_states[2]:
            blue_score += 1

        # Count pole points
        for pole_row in self._pole_states:
            for pole in pole_row:
                # Sum points scored by each alliance on the pole
                if pole.pole_type == "ground":
                    red_score += pole.red_scored * 2
                    blue_score += pole.blue_scored * 2

                elif pole.pole_type == "low":
                    red_score += pole.red_scored * 3
                    blue_score += pole.blue_scored * 3

                elif pole.pole_type == "mid":
                    red_score += pole.red_scored * 4
                    blue_score += pole.blue_scored * 4

                elif pole.pole_type == "high":
                    red_score += pole.red_scored * 5
                    blue_score += pole.blue_scored * 5

                # Calculate owning bonuses
                if pole.pole_owner == "red":
                    red_score += 3
                elif pole.pole_owner == "blue":
                    blue_score += 3

        return red_score, blue_score

    def _set_pole_states(self):
        self._pole_states[0][0].pole_type = "ground"
        self._pole_states[0][2].pole_type = "ground"
        self._pole_states[0][4].pole_type = "ground"
        self._pole_states[2][0].pole_type = "ground"
        self._pole_states[2][2].pole_type = "ground"
        self._pole_states[2][4].pole_type = "ground"
        self._pole_states[4][0].pole_type = "ground"
        self._pole_states[4][2].pole_type = "ground"
        self._pole_states[4][4].pole_type = "ground"

        self._pole_states[0][1].pole_type = "low"
        self._pole_states[0][3].pole_type = "low"
        self._pole_states[1][0].pole_type = "low"
        self._pole_states[1][4].pole_type = "low"
        self._pole_states[3][0].pole_type = "low"
        self._pole_states[3][4].pole_type = "low"
        self._pole_states[4][1].pole_type = "low"
        self._pole_states[4][3].pole_type = "low"

        self._pole_states[1][1].pole_type = "mid"
        self._pole_states[1][3].pole_type = "mid"
        self._pole_states[3][1].pole_type = "mid"
        self._pole_states[3][3].pole_type = "mid"

        self._pole_states[1][2].pole_type = "high"
        self._pole_states[2][1].pole_type = "high"
        self._pole_states[2][3].pole_type = "high"
        self._pole_states[3][2].pole_type = "high"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Times the matches
        self._step_count = 0
        self._end_game = False

        self._agent_red_carrying = False
        self._agent_blue_carrying = False

        self._pole_states = [[PoleState()] * 5] * 5
        self._set_pole_states()

        self._corner_states = [False] * 4

        self._agent_red_location = np.array([0, 0])
        self._agent_blue_location = np.array([5, 5])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # MOVE

        # If we want to move agent red
        if action < 4:
            direction = self._action_to_direction[action]
            self._agent_red_location = np.clip(self._agent_red_location + direction, 0, self.size - 1)

        # If we want to move agent blue
        elif action < 8:
            direction = self._action_to_direction[action - 4]
            self._agent_blue_location = np.clip(self._agent_blue_location + direction, 0, self.size - 1)

        # DEPOSIT

        # If we want red to deposit and agent red is carrying something
        elif action < 12 and self._agent_red_carrying:
            redef_action = (action - 12) + 4

            # Deposit NE
            if redef_action == 0:
                deposit_r = self._agent_red_location[0] - 1
                deposit_c = self._agent_red_location[1]
                self._score(deposit_r, deposit_c, "red")
                self._agent_red_carrying = False

            # Deposit NW
            elif redef_action == 1:
                deposit_r = self._agent_red_location[0] - 1
                deposit_c = self._agent_red_location[1] - 1
                self._score(deposit_r, deposit_c, "red")
                self._agent_red_carrying = False

            # Deposit SW
            elif redef_action == 2:
                deposit_r = self._agent_red_location[0]
                deposit_c = self._agent_red_location[1] - 1
                self._score(deposit_r, deposit_c, "red")
                self._agent_red_carrying = False

            # Deposit SE
            elif redef_action == 3:
                self._score(*self._agent_red_location, "red")
                self._agent_red_carrying = False

        # If we want blue to deposit and agent blue is carrying something
        elif action < 16 and self._agent_blue_carrying:
            redef_action = (action - 16) + 4

            # Deposit NE
            if redef_action == 0:
                deposit_r = self._agent_blue_location[0] - 1
                deposit_c = self._agent_blue_location[1]
                self._score(deposit_r, deposit_c, "blue")
                self._agent_blue_carrying = False

            # Deposit NW
            elif redef_action == 1:
                deposit_r = self._agent_blue_location[0] - 1
                deposit_c = self._agent_blue_location[1] - 1
                self._score(deposit_r, deposit_c, "blue")
                self._agent_blue_carrying = False

            # Deposit SW
            elif redef_action == 2:
                deposit_r = self._agent_blue_location[0]
                deposit_c = self._agent_blue_location[1] - 1
                self._score(deposit_r, deposit_c, "blue")
                self._agent_blue_carrying = False

            # Deposit SE
            elif redef_action == 3:
                self._score(*self._agent_blue_location, "blue")
                self._agent_blue_carrying = False

        # INTAKE

        # Agent red intake - can only if on the correct position
        elif action == 16:
            if np.array_equal(self._agent_red_location, [2, 5]) or np.array_equal(self._agent_red_location, [3, 5]):
                self._agent_red_carrying = True

        # Agent blue intake - can only if on the correct position
        elif action == 17:
            if np.array_equal(self._agent_blue_location, [2, 0]) or np.array_equal(self._agent_blue_location, [3, 0]):
                self._agent_blue_carrying = True

        # CORNER DEPOSIT

        # Agent blue deposit on NE corner
        elif action == 18:
            if np.array_equal(self._agent_blue_location, [0, 5]) and self._agent_blue_carrying:
                self._corner_states[0] = True
                self._agent_blue_carrying = False

        # Agent red deposit on NW corner
        elif action == 19:
            if np.array_equal(self._agent_red_location, [0, 0]) and self._agent_red_carrying:
                self._corner_states[1] = True
                self._agent_red_carrying = False

        # Agent blue deposit on SW corner
        elif action == 20:
            if np.array_equal(self._agent_blue_location, [5, 0]) and self._agent_blue_carrying:
                self._corner_states[2] = True
                self._agent_blue_carrying = False

        # Agent red deposit on SE corner
        elif action == 21:
            if np.array_equal(self._agent_red_location, [5, 5]) and self._agent_red_carrying:
                self._corner_states[3] = True
                self._agent_red_carrying = False

        # Take care of returns

        # Terminate based on time into match
        terminated = self._step_count > 90

        # Set the reward to purely the reward
        # TODO: Validate if this reward function is good
        reward_red, reward_blue = self._calculate_score()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._step_count += 1

        # Important: returns the rewards as a tuple with red reward first
        return observation, (reward_red, reward_blue), terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)  # The size of a single grid square in pixels

        # Draw agent red
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * np.flip(self._agent_red_location),
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw agent blue
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                pix_square_size * np.flip(self._agent_blue_location),
                (pix_square_size, pix_square_size),
            ),
        )

        # Add grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
