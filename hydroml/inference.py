import numpy as np
from torch import load
from torch.utils.data import DataLoader

def run(data_path, gridNx, model_name):
    net_Baryons = np.fromfile(data_path, dtype=np.float32)
    net_Baryons = net_Baryons.reshape(net_Baryons.size // gridNx, gridNx)

    eta = net_Baryons[0]
    data = net_Baryons[1:].reshape((net_Baryons.size - 1) // gridNx, 1, gridNx)

    print(data.shape)

    model = load(f'models/{model_name}')
    model.eval()

    data_loader = DataLoader(
        dataset=data,
        batch_size=1,
        shuffle=False
    )

    model_outputs = []

    for i, data in enumerate(data_loader):
        key = data

        protons_model = model(key)

        model_outputs.append(protons_model.detach().numpy().flatten())

    np.array(model_outputs, dtype=np.float32).tofile(f'{data_path[:-4]}_gen_netProton.dat')