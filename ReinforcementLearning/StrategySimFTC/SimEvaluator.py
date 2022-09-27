import ftc_field
import gym

env = gym.make('ftc_field/ftc_field-v0', render_mode='human')

obs = env.reset()

terminated = False

# while not terminated:
#     action = 1
#     if type(obs) == dict:
#         print("Agent red pos: ", obs["agent_red"], "Agent red carrying", obs["carrying"])
#     obs, reward, terminated, info = env.step(action)

# env.step(3)

env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(0)
env.step(3)
env.step(3)
obs, reward, terminated, info = env.step(16)
print("Agent red pos: ", obs["agent_red"], "Agent red carrying", obs["carrying"][0])
env.step(2)
env.step(2)
env.step(8)
# TODO: Carrying information is not being updated correctly.
print("Agent red pos: ", obs["agent_red"], "Agent red carrying", obs["carrying"][0])

env.close()
