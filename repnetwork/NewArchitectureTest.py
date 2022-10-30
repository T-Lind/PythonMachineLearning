import tensorflow as tf

import repnet.core
from repnet.LogData import print_compile_data
from repnet.examples.Cartpole import CartPoleEnv

if __name__ == '__main__':
    num_networks_train = 10

    reward_list = []
    for i in range(num_networks_train):
        reward_vals, _ = repnet.core.train(
            model=repnet.ActorCritic(2, 128),
            env_obj=CartPoleEnv(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            tree=repnet.core.Tree(100, lambda x: 100),
            running_reward_threshold=195,
        )
        print(len(reward_vals))
        reward_list.append(reward_vals)
    print_compile_data(reward_list)
