import numpy as np
from numba import jit
from torch.utils.data import Dataset

@jit
def _kernel(x, x0):
    sigma = 0.8
    protonFraction = 0.4
    norm = protonFraction / (np.sqrt(2. * np.pi) * sigma)
    return norm * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

@jit
def _fake_kernel(x, x0, s):
    sigma = s
    protonFraction = 0.4
    norm = protonFraction / (np.sqrt(2. * np.pi) * sigma)
    return norm * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

@jit
def _test_data_gen(fakekernel=False, sigma=0.4):
    A = 197
    yBeam = 5.36
    slope = 0.5
    sigmaEtas = 0.2

    # generate input data
    nBaryons = np.random.randint(0, 2 * A)
    randX = np.random.uniform(0, 1, size=nBaryons)
    etasBaryon = 1. / slope * np.arcsinh((2. * randX - 1) * np.sinh(slope * yBeam))
    etasArr = np.linspace(-6.4, 6.4, 128)
    dNBdetas = np.zeros(len(etasArr))
    norm = 1. / (np.sqrt(2. * np.pi) * sigmaEtas)
    for iB in etasBaryon:
        dNBdetas += norm * np.exp(-(etasArr - iB) ** 2. / (2. * sigmaEtas ** 2.))

    # generate test data with convolution with a kernel
    dNpdy = np.zeros(len(etasArr))
    detas = etasArr[1] - etasArr[0]
    for i in range(len(etasArr)):
        dNpdy[i] = sum(_kernel(etasArr, etasArr[i]) * dNBdetas) * detas

    return etasArr, dNBdetas, dNpdy

def generate_data(size):
    real_data = []

    for iev in range(size):
        x, y1, y2 = _test_data_gen()

        # real data - Block
        x = y2
        real_data.append(x)

    return np.array(real_data, dtype=np.float32)

class Data(Dataset):
    def __init__(self, size=1):
        assert(size >= 1)
        self.size = size
        self.data = generate_data(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.data[item]