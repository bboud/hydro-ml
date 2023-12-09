import numpy as np
import torch as t
from torch.utils.data import DataLoader

def run(data, model_path):
    model = t.load(f'{model_path}')
    model.eval()

    data_loader = DataLoader(
        dataset=data,
        batch_size=1,
        shuffle=False
    )

    model_outputs = []

    for _, d in enumerate(data_loader):
        model_outputs.append(model(d).detach().numpy().flatten())

    np.save("output", model_outputs, allow_pickle=True)