import numpy as np
from torch.utils.data import Dataset
import torch

import h5py

# Warning!
# This dataset class is designed for the .dat files.
# This includes the first 141 elements being eta
# and all the rest being the data..

# Different import methods will require a different Dataset class

class TrainDataset(Dataset):
    def __init__(self, keys: np.ndarray, values: np.ndarray, gridNx):
        self.gridNx = gridNx

        # Reshape the data to (Number of events, Number of datapoints)
        keys = keys.reshape(keys.size // gridNx, gridNx)
        self.eta = keys[0]
        self.keys = keys[1:]

        self.kshape = self.keys.shape

        if values is not None:
            values = values.reshape(values.size // gridNx, gridNx)
            self.values = values[1:]
            self.vshape = self.keys.shape

        super(Dataset, self).__init__()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        if self.values is None:
            return self.keys[item], np.zeros_like(self.eta)
        else:
            return self.keys[item], self.values[item]

    def __add__(self, other):
        self.keys = np.append(self.keys, other.keys)
        self.keys = self.keys.reshape(self.keys.size // self.gridNx, self.gridNx)
        if self.values is not None:
            self.values = np.append(self.values, other.values)
            self.values = self.values.reshape(self.values.size // self.gridNx, self.gridNx)

        return self
    
class InferenceDataset(Dataset):
    def __init__(self, keys: np.ndarray, gridNx):
        self.gridNx = gridNx

        # Reshape the data to (Number of events, Number of datapoints)
        keys = keys.reshape(keys.size // gridNx, gridNx)
        self.eta = keys[0]
        self.keys = keys[1:]

        self.kshape = self.keys.shape

        super(Dataset, self).__init__()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.keys[item]

    def __add__(self, other):
        self.keys = np.append(self.keys, other.keys)
        self.keys = self.keys.reshape(self.keys.size // self.gridNx, self.gridNx)

        return self
    
class VectorsDataset(Dataset):
    def __init__(self, file):
        super(Dataset, self).__init__()

        self.results = []

        data = h5py.File(file)

        def _getInitalCompTensor(result):
            eccent_list = []
            for i in result.keys():
                if "eccentricities" in i:
                    eccent_list.append(i)

            # Get the last ed_tau
            initial = result[eccent_list[-1]]
            comp_tens = torch.transpose(torch.tensor(initial[:], dtype=torch.float32), 0, 1)
            initial_tensor = [comp_tens[0], comp_tens[4], comp_tens[5]]
            return initial_tensor
                
        def _getFinalCompTensor(result):
            final = result['particle_9999_dNdeta_pT_0.2_3.dat']
            comp_tens = torch.transpose(torch.tensor(final[:], dtype=torch.float32), 0, 1)
            final_tensor = [comp_tens[0], comp_tens[5], comp_tens[6]]
            return final_tensor

        for v in data.keys():
            spvn_result = data[v]

            self.results.append(
                (_getInitalCompTensor(spvn_result), _getFinalCompTensor(spvn_result))
            )

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        return (self.results[item])