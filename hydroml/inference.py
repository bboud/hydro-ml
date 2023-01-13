import matplotlib.pyplot as plt
import numpy as np
from torch import load
from torch.utils.data import DataLoader

from hydroml.libs.dataset import Dataset

def run(data_path, gridNx):
    print(data_path)
    net_Baryons = Dataset(np.fromfile(data_path, dtype=np.float32), gridNx)

    model = load('models/test_19_L.pt')
    model.eval()

    data_loader = DataLoader(
        dataset=net_Baryons,
        shuffle=False
    )

    model_outputs = []

    for i, data in enumerate(data_loader):
        key = data.flatten()

        protons_model = model(key)

        model_outputs.append(protons_model.detach().numpy().flatten())

    np.array(model_outputs, dtype=np.float32).tofile(f'{data_path}_gen_netProton.dat')

    verify()

def verify():
    net_Protons = Dataset(np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.2_wBulk_22momdeltaf_netProton.dat', dtype=np.float32), 141)
    net_Protons_gen = Dataset(np.fromfile('datasets/training/NetbaryonDis_OSG3DAuAu19.6_tune18.2_wBulk_22momdeltaf_netBaryon.dat_gen_netProton.dat', dtype=np.float32), 141)

    for i in range(25, 30):
        plt.plot(net_Protons.eta, net_Protons.data[i])
        plt.plot(net_Protons.eta, net_Protons_gen.data[i])
        plt.show()