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

net_Baryons_19_1 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.3_wBulk_22momdeltaf_netBaryon.dat', dtype=np.float32)
net_Protons_19_1 = np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.3_wBulk_22momdeltaf_netProton.dat', dtype=np.float32)

net_Baryons_19_1 = net_Baryons_19_1.reshape( (len(net_Baryons_19_1) // 141, 141) )
net_Protons_19_1 = net_Protons_19_1.reshape( (len(net_Protons_19_1) // 141, 141) )[1:]

eta = net_Baryons_19_1[0]
net_Baryons_19_1 = net_Baryons_19_1[1:]

# Last argument can be adapted to change model resolution
dataset_1 = TrainDataset(net_Baryons_19_1, net_Protons_19_1, eta, 3.5)

data_loader = DataLoader(
    dataset=dataset_1,
    batch_size=1,
    shuffle=True,
)

print(f'Training on {len(data_loader)} samples')

model = BaryonModel(len(dataset_1.eta), len(dataset_1.eta))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_func = torch.nn.MSELoss()

for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    for i, data in enumerate(data_loader):
        keys = data[0]
        values = data[1]

        output = model(keys)

        optimizer.zero_grad()

        loss = loss_func(output, values)

        loss.backward()
        optimizer.step()

torch.save(model, "models/baryon_model_19.6gev.pt")