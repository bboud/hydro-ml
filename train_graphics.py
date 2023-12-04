import sys

import matplotlib.pyplot as plt
import numpy as np

import os
sys.path.insert(0, os.path.abspath(''))

from hydroml.model import BaryonModel

from torch.utils.data import DataLoader

from torch import nn

import torch
#%%
batch_size = 1
ngpu = 0
# Epochs set to '1' for testing
epochs = 1
learning_rate = 1e-3
beta1 = 0.001
n_samples = 20000

from hydroml.dataset import Dataset

net_Baryons_19 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.2_wBulk_22momdeltaf_netBaryon.dat', dtype=np.float32)

net_Protons_19 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.2_wBulk_22momdeltaf_netProton.dat', dtype=np.float32)

net_Baryons_19_2 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.3_wBulk_22momdeltaf_netBaryon.dat', dtype=np.float32)

net_Protons_19_2 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.3_wBulk_22momdeltaf_netProton.dat', dtype=np.float32)

net_Baryons_200 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu200_tune18.6_wBulk_22momdeltaf_wHBT_netBaryon.dat', dtype=np.float32)

net_Protons_200 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu200_tune18.6_wBulk_22momdeltaf_wHBT_netProton.dat', dtype=np.float32)

dataset2 = Dataset(net_Baryons_19, net_Protons_19, 141)
dataset3 = Dataset(net_Baryons_19_2, net_Protons_19_2, 141)
dataset1 = Dataset(net_Baryons_200, net_Protons_200, 141)

dataset = dataset1 + dataset2 + dataset3

data_loader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=True,
)
#%%
model = BaryonModel()
#%%
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
#%%
loss_func = nn.MSELoss()

###############Primary Training Loop###############

for i, data in enumerate(data_loader):
    keys = data[0]
    values = data[1]

    optimizer.zero_grad()

    output = model(keys)

    loss = loss_func(output, values)

    loss.backward()
    optimizer.step()

    if (i<500): #and output.detach().numpy()[0].flatten().max() > 10:
        print(output.detach().numpy()[0].flatten().max())
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot()
        ax.plot(dataset.eta, values[0].numpy().flatten(), label="Real Final State", color='blue')
        ax.plot(dataset.eta, output.detach().numpy()[0].flatten(), label="Generated Final State", color='green')

        ax.set_title("Final State Net Proton Pseudorapidity Distribution")
        ax.set_ylabel("$dN^{ch}/d\eta$")
        ax.set_xlabel("$\eta$")
        ax.legend(loc = "upper left")
        plt.savefig(f"frames2/Frame{i}")
        plt.close(fig)

    if i >= 1000:
        break
