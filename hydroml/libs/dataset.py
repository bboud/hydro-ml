import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data: np.ndarray, gridNx):
        self.gridNx = gridNx

        # Reshape the data to (Number of events, Number of Columns, Number of datapoints)
        data = data.reshape(data.size // gridNx, gridNx)

        self.eta = data[0]
        self.data = data[1:]

        super(Dataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]