import numpy as np
from torch.utils.data import Dataset
import torch
from hydroml.utils import batch_poly_regression, batch_trim

import h5py

class TrainDataset(Dataset):
    def __init__(self, keys: np.ndarray, values: np.ndarray, eta, resolution=None):

        if resolution is None:
            self.eta = eta
            self.keys = keys
            self.values = batch_poly_regression(self.eta, values, 15 ) or np.ones_like(self.eta)
            
            super(Dataset, self).__init__()
            return

        self.eta, self.keys = batch_trim(eta, keys, -resolution, resolution)

        if values is not None:
            _, self.values = batch_trim(eta, values, -resolution, resolution)
            self.values = batch_poly_regression(self.eta, self.values, 15 )

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
        self.keys = self.keys.reshape(self.keys.size // len(self.eta), len(self.eta))
        if self.values is not None:
            self.values = np.append(self.values, other.values)
            self.values = self.keys.reshape(self.keys.size // len(self.eta), len(self.eta))

        return self
    
class TrainDatasetRu(Dataset):
    def __init__(self, keys: np.ndarray, values: np.ndarray, etaInit, etaFinal, sizeInit, sizeFinal, resolution=None):

        if resolution is None:
            self.etaInit = etaInit
            self.etaFinal = etaFinal
            self.keys = keys
            self.values = batch_poly_regression(self.etaFinal, values, 15 ) or np.ones_like(self.eta)
            
            super(Dataset, self).__init__()
            return

        self.etaInit, self.keys = batch_trim(etaInit, keys, -resolution, resolution)

        if values is not None:
            self.etaFinal, self.values = batch_trim(etaFinal, values, -resolution, resolution)
            self.values = batch_poly_regression(self.etaFinal, self.values, 15 )

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
        self.keys = self.keys.reshape(self.keys.size // len(self.etaInit), len(self.etaInit))
        if self.values is not None:
            self.values = np.append(self.values, other.values)
            self.values = self.keys.reshape(self.keys.size // len(self.etaFinal), len(self.etaFinal))

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