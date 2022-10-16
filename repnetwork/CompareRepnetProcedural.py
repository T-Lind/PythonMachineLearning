from repnet import train_repnet, train_rl, train_distributed_repnet
import matplotlib.pyplot as plt
import numpy as np

# Number of each network to train and then average
num_networks_trained = 10

# Lists that will store the running reward data across episodes
repnet_performances = []
dist_repnet_performances = []
rl_performances = []

# The loop to train each of the three types of networks num_networks_trained amount of times
for i in range(num_networks_trained):
    if i % 10 == 0:
        print(f"{100*i/num_networks_trained}% done!")

    # Different hyperparameters can be set using kwargs
    repnet_perform = train_repnet()
    dist_repnet_perform = train_distributed_repnet()
    rl_perform = train_rl()

    # Not recommended to plot a large number of networks at once
    # plt.plot(repnet_perform, color="magenta")
    # plt.plot(dist_repnet_perform, color="black", label="Distributed REPNET)
    # plt.plot(rl_perform, color="teal", label="Normal RL")

    rl_performances.append(rl_perform)
    repnet_performances.append(repnet_perform)
    dist_repnet_performances.append(dist_repnet_perform)


if __name__ == '__main__':
    # Get the number of episodes for each training instance each network required to achieve the threshold (195)
    repnet_data = [len(x) for x in repnet_performances]
    dist_repnet_data = [len(x) for x in dist_repnet_performances]
    rl_data = [len(x) for x in rl_performances]

    # Print the mean and standard deviation of each network across num_networks_trained number of each network
    print(f"REPNET average # of weight updates: {np.mean(repnet_data)}, standard dev: {np.std(repnet_data)}")
    print(f"Distributed REPNET average # of weight updates: {np.mean(repnet_data)}, standard dev: {np.std(repnet_data)}")
    print(f"RL average # of weight updates {np.mean(rl_data)}, standard dev: {np.std(rl_data)}")
    #
    # plt.title("REPNET performance versus normal actor-critic RL on the CartPole env:")
    # plt.xlabel("Running reward")
    # plt.ylabel("Cumulative training iterations")
    # plt.show()
