import sys
import numpy as np

sys.path.append('.')

from hydroml.model import BaryonModel
from hydroml.dataset import TrainDataset

from torch.utils.data import DataLoader

import torch

batch_size = 1
epochs = 20
learning_rate = 1e-5

net_Baryons_19_1 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.2_wBulk_22momdeltaf_netBaryon.dat', dtype=np.float32)
net_Protons_19_1 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.2_wBulk_22momdeltaf_netProton.dat', dtype=np.float32)
net_Baryons_19_2 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.3_wBulk_22momdeltaf_netBaryon.dat', dtype=np.float32)
net_Protons_19_2 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.3_wBulk_22momdeltaf_netProton.dat', dtype=np.float32)

dataset_1 = TrainDataset(net_Baryons_19_1, net_Protons_19_1, 141)
dataset_2 = TrainDataset(net_Baryons_19_2, net_Protons_19_2, 141)
dataset = dataset_1 + dataset_2

data_loader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=True,
)

model = BaryonModel()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_func = torch.nn.MSELoss(reduction='sum')

for epoch in range(epochs):
    for i, data in enumerate(data_loader):
        keys = data[0]
        values = data[1]

        output = model(keys)

        optimizer.zero_grad()

        loss = loss_func(output, values)

        loss.backward()
        optimizer.step()

torch.save(model, "models/baryon_model_19.6gev.pt")