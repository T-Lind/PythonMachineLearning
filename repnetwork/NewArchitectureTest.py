import tensorflow as tf

from repnet import ActorCritic
from repnet.core import Repnet, Tree
from repnet.examples.Cartpole import CartPoleEnv

cartpole = CartPoleEnv()
tree_structure = Tree(50, lambda x: 75)
actor_critic = ActorCritic(cartpole.env.action_space.n, 128)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
repnet = Repnet(tree=tree_structure, model=actor_critic, optimizer=optimizer)

data = repnet.train(cartpole)
print(data)
