import ftc_field
import gym

env = gym.make('ftc_field/ftc_field-v0', render_mode='human')

obs = env.reset()

terminated = False

while not terminated:
    action = 3
    if type(obs) == dict:
        print("Agent red pos: ", obs["agent_red"], "Agent red carrying", obs["carrying"][0])
    obs, reward, terminated, info = env.step(action)

env.close()
