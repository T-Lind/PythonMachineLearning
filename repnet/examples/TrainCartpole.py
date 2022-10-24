from repnet.examples.CartpoleTrainingFunctions import train_repnet, train_distributed_repnet, train_rl
import numpy as np
import matplotlib.pyplot as plt

num_networks_trained = 1

# Lists that will store the running reward data across episodes
repnet_performances = []
dist_repnet_performances = []
rl_performances = []

#The loop to train each of the three types of networks num_networks_trained amount of times
for i in range(num_networks_trained + 1):
    if i % 10 == 0 and i > 0:
        print(f"{100 * i / num_networks_trained}% done!")
        repnet_data = [len(x) for x in repnet_performances]
        dist_repnet_data = [len(x) for x in dist_repnet_performances]
        rl_data = [len(x) for x in rl_performances]

        # Print the mean and standard deviation of each network across num_networks_trained number of each network
        print(f"REPNET average # of weight updates: {np.mean(repnet_data)}, standard dev: {np.std(repnet_data)}")
        print(
            f"Distributed REPNET average # of weight updates: {np.mean(dist_repnet_data)}, standard dev: {np.std(dist_repnet_data)}")
        print(f"RL average # of weight updates {np.mean(rl_data)}, standard dev: {np.std(rl_data)}")

    # Different hyperparameters can be set using kwargs
    repnet_perform = train_repnet()
    dist_repnet_perform = train_distributed_repnet(threshold_func=lambda x: 15, kill_time=75)
    rl_perform = train_rl()

    # Not recommended to plot a large number of networks at once
    if i == 1:
      # Only use the labels the first time
      plt.plot(repnet_perform, color="magenta", label="REPNET")
      plt.plot(dist_repnet_perform, color="black", label="Distributed REPNET")
      plt.plot(rl_perform, color="teal", label="Normal RL")
    else:
      plt.plot(repnet_perform, color="magenta")
      plt.plot(dist_repnet_perform, color="black")
      plt.plot(rl_perform, color="teal")

    rl_performances.append(rl_perform)
    repnet_performances.append(repnet_perform)
    dist_repnet_performances.append(dist_repnet_perform)

# Get the number of episodes for each training instance each network required to achieve the threshold (195)
repnet_data = [len(x) for x in repnet_performances]
dist_repnet_data = [len(x) for x in dist_repnet_performances]
rl_data = [len(x) for x in rl_performances]

# Print the mean and standard deviation of each network across num_networks_trained number of each network
print(f"REPNET average # of weight updates: {np.mean(repnet_data)}, standard dev: {np.std(repnet_data)}")
print(
    f"Distributed REPNET average # of weight updates: {np.mean(dist_repnet_data)}, standard dev: {np.std(dist_repnet_data)}")
print(f"RL average # of weight updates {np.mean(rl_data)}, standard dev: {np.std(rl_data)}")

plt.title("REPNET versus normal actor-critic RL on the CartPole env:")
# plt.title("REPNET and distributed variant versus normal actor-critic RL on the CartPole env:")
plt.xlabel("Running reward")
plt.ylabel("Cumulative training episodes")
plt.legend()
plt.show()
