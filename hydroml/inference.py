import matplotlib.pyplot as plt

from hydroml.libs.dataset import Dataset

def run(data_path, gridNx):
    dataset = Dataset(data_path, gridNx)

    plt.plot(dataset.eta, dataset.data[5])
    plt.show()