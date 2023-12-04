import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):
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
            return self.keys[item], None
        else:
            return self.keys[item], self.values[item]

    def __add__(self, other):
        self.keys = np.append(self.keys, other.keys)
        self.keys = self.keys.reshape(self.keys.size // self.gridNx, self.gridNx)
        if self.values is not None:
            self.values = np.append(self.values, other.values)
            self.values = self.values.reshape(self.values.size // self.gridNx, self.gridNx)
        return self