from torch import load
from torch.utils.data import DataLoader

def single_inference(data, model_path):
    print(data.shape)

    model = load(f'{model_path}')
    model.eval()

    data_loader = DataLoader(
        dataset=data,
        batch_size=1,
        shuffle=False
    )

    for _, data in enumerate(data_loader):
        key = data

        protons_model = model(key)

        return protons_model.detach().numpy().flatten()