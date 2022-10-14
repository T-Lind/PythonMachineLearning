from repnet import train_repnet, train_rl, train_rl_hardstop
import matplotlib.pyplot as plt
import numpy as np

num_networks_trained = 10

repnet_performances = []
rl_performances = []

for i in range(num_networks_trained):
    if i % 10 == 0:
        print(f"{100*i/num_networks_trained}% done!")

    repnet_perform = train_repnet()

    repnet_performances.append(repnet_perform)

num_nets_rl = num_networks_trained
while num_nets_rl > 0:
    rl_perform = train_rl_hardstop()
    if len(rl_perform) > 1:
        num_nets_rl -= 1

    rl_performances.append(rl_perform)

    # plt.plot(repnet_perform, color="magenta", label="REPNET")
    # plt.plot(rl_perform, color="teal", label="Normal RL")

repnet_data = [len(x) for x in repnet_performances]
rl_data = [len(x) for x in rl_performances]
print(f"REPNET average # of weight updates: {np.mean(repnet_data)}, standard dev: {np.std(repnet_data)}")
print(f"RL average # of weight updates with a hardstop at 300: {np.mean(rl_data)}, standard dev: {np.std(rl_data)}")
#
# plt.title("REPNET performance versus normal actor-critic RL on the CartPole env:")
# plt.xlabel("Running reward")
# plt.ylabel("Cumulative training iterations")
# plt.show()
