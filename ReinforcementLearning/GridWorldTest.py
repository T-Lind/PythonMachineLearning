import gym_examples
import gym

import gym

env = gym.make('gym_examples/GridWorld-v0', render_mode='human', size=10)


def basic_policy(obs):
    if type(obs) == dict:
        agent_r = obs['agent'][0]
        agent_c = obs['agent'][1]
        target_r = obs['target'][0]
        target_c = obs['target'][1]

        if agent_r < target_r:
            return 0
        if agent_r > target_r:
            return 2
        if agent_c < target_c:
            return 1
        if agent_c > target_c:
            return 3

    return 0


obs = env.reset()

terminated = False
while not terminated:
    action = basic_policy(obs)
    obs, reward, terminated, info = env.step(action)
env.close()
