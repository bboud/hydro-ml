import numpy as np
from torch.utils.data import Dataset

class InfDataset(Dataset):
    def __init__(self, keys: np.ndarray, values: np.ndarray, gridNx):
        self.gridNx = gridNx

        # Reshape the data to (Number of events, Number of Columns, Number of datapoints)
        keys = keys.reshape(keys.size // gridNx, gridNx)
        values = values.reshape(values.size // gridNx, gridNx)

        self.eta = keys[0]
        self.keys = keys[1:]

        self.values = values[1:]

        super(InfDataset, self).__init__()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.keys[item], self.values[item]