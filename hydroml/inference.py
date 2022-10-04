import sys

from torch.utils.data import DataLoader

sys.path.append('../hydroml')

from torch import load
from hydroml.dataset import EnergyDensityDataset

model = load('../Trained Models/baryon_model_19gev.pt')
model.eval()


dataset = EnergyDensityDataset('../Datasets/dE/dE_data_1/dE_detas_initial', '../Datasets/dE/dE_data_1/dET_deta_final').smooth().cosh().trim(-6.8,8).remove_anamalies(150)

data_loader = DataLoader(
    dataset=dataset,
    shuffle=True
)