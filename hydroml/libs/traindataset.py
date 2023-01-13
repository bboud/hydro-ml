import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, keys: np.ndarray, values: np.ndarray, gridNx):
        self.gridNx = gridNx

        # Reshape the data to (Number of events, Number of Columns, Number of datapoints)
        keys = keys.reshape(keys.size // gridNx, gridNx)
        values = values.reshape(values.size // gridNx, gridNx)

        self.eta = keys[0]
        self.keys = keys[1:]

        self.values = values[1:]

        self.add_noise()
        self.high_energy_filter(100.0)

        super(TrainDataset, self).__init__()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.keys[item], self.values[item]

    def remove_item(self, i):
        self.keys = np.delete(self.keys, i, 0)
        self.values = np.delete(self.values, i, 0)

    def add_noise(self):
        self.keys = np.array(self.keys * np.random.normal(0.9, 0.05, (len(self.keys), 141)), dtype=np.float32)

    def low_energy_filter(self, factor):
        indicies = []
        for i, data in enumerate(self.values):
            if data.max() < factor:
                indicies.append(i)

        self.remove_item(indicies)

    def high_energy_filter(self, factor):
        indicies = []
        for i, data in enumerate(self.values):
            if data.max() > factor:
                indicies.append(i)

        self.remove_item(indicies)

    def smooth(self):
        for i, data in enumerate(self.keys):
            for j in range(2, len(data) - 2):
                average = np.float64((data[j - 2] + data[j - 1] + data[j] + data[j + 1] + data[j + 2]) / 5)
                data[j] = average
            self.keys[i] = data
        for i, data in enumerate(self.values):
            for j in range(2, len(data) - 2):
                average = np.float64((data[j - 2] + data[j - 1] + data[j] + data[j + 1] + data[j + 2]) / 5)
                data[j] = average
            self.values[i] = data