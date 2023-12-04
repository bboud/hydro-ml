import torch as t
from torch.utils.data import DataLoader

def single_inference(data, model_path):

    model = t.load(f'{model_path}')
    model.eval()

    return model(t.tensor(data, dtype=t.float32)).detach().numpy().flatten()