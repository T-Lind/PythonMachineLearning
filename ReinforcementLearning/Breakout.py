import keyboard
from tf_agents.environments import suite_gym

env = suite_gym.load('Breakout-v4', render_mode='human')


obs = env.reset()

for episode in range(10):
    obs = env.reset()
    for step in range(200):
        if keyboard.is_pressed('q'):
            exit()
