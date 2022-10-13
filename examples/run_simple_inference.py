#Note: In order to import the files for the hydroml model, you will need to correctly point to the hydroml directory. Paths are relative.
import sys
sys.path.append('../hydroml')

import inference as inf
from dataset import EnergyDensityDataset

import matplotlib.pyplot as plt
import torch

#Create and load the trained model
model = torch.load('../models/dE_model.pt')
model.eval()

#It is recommended to use the dataset loading feature because it comes with the functions to manupulate the data, however, the inference function does not care
# what you give it as long as it is of the correct size (462).
dataset = EnergyDensityDataset('../data/dE/dE_data_1/dE_detas_initial', '../data/dE/dE_data_1/dET_deta_final').smooth().cosh().trim(-6.8,8).remove_anomalies(150)

#If you use the dataset class to load the data, the fisrt index will reference the event. The second index will be either 0 or 1.
# 0 - initial state distribution
# 1 - final state distribution
# So this next line will give you the 0'th event's initial state distribution.
initial = dataset[0][0]
output = inf.inference(model, initial)

#To access the eta values from the dataset, you can use dataset.start_eta and dataset.final_eta.
# These are arrays that contain the respective x-axis.

# A smoothing function has been included so remove jitter from the model's output. This may or may not be desirable.
plt.plot(dataset.final_eta, inf.smooth(output))
plt.plot(dataset.start_eta, initial)
plt.show()