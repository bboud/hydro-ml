import numpy
import numpy as np
import os
import sys
from numba import jit
from torch.utils.data import Dataset

######################################## TEST DATA GENERATION ########################################

@jit
def _kernel(x, x0):
    if x0 >= 0:
        sigma = 0.9
    if x0 < 0:
        sigma = 0.1
    protonFraction = 0.4
    norm = protonFraction / (np.sqrt(2. * np.pi) * sigma)
    return norm * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

@jit
def _test_data_gen():
    A = 197
    yBeam = 5.36
    slope = 0.5
    sigmaEtas = 0.2

    # generate input data
    nBaryons = np.random.randint(0, 2 * A)
    randX = np.random.uniform(0, 1, size=nBaryons)
    etasBaryon = 1. / slope * np.arcsinh((2. * randX - 1) * np.sinh(slope * yBeam))
    etasArr = np.linspace(-6.4, 6.4, 141)
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
    baryons = []
    protons = []

    for iev in range(size):
        x, y1, y2 = _test_data_gen()

        baryons.append(np.array(y1).reshape(1,141))
        protons.append(np.array(y2).reshape(1,141))

    return np.array(baryons, dtype=np.float32), np.array(protons, dtype=np.float32)

class GenData(Dataset):
    def __init__(self, size=1):
        assert(size >= 1)
        self.size = size
        self.data, self.labels = generate_data(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

######################################## REAL DATA IMPORT ########################################

def get_real_data(dataset, size):
    events = []
    i = 0
    for file in os.listdir(f'./{dataset}/'):
        if i >= size:
            break
        # Only register events with baryon_etas
        if file.find('baryon_etas') == -1:
            continue
        # Grab all event numbers
        events.append(file.split('_')[1])
        i += 1

    baryons = []
    protons = []

    for event in events:
        # Eta is the same for the datasets
        eta_baryon, baryon = np.loadtxt(
            f'./{dataset}/event_{event}_net_baryon_etas.txt', unpack=True)
        eta_proton, proton, error = np.loadtxt(
            f'./{dataset}/event_{event}_net_proton_eta.txt', unpack=True)

        #Low energy filter
        if proton.max() < 1.:
            continue

        baryon = data_smoothing(baryon)
        proton = data_smoothing(data_smoothing(proton))

        baryons.append(baryon.reshape(1, 141))
        protons.append(proton.reshape(1, 141))

    return np.array(baryons, dtype=np.float32), np.array(protons, dtype=np.float32), eta_proton

def data_smoothing(data):
    # Calculate the moving average
    for i in range( 2, len(data) - 2 ):
        average = np.float64( (data[i-2]  + data[i-1] + data[i] + data[i+1] + data[i+2]) / 5 )
        data[i] = average
    return data

class Data(Dataset):
    def __init__(self, dataset, size=sys.maxsize):
        assert(size >= 1)
        self.data, self.labels, self.data_axis = get_real_data(dataset, size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    # Using multiple datasets, add them together
    def __add__(self, other):
        self.data = numpy.concatenate((self.data, other.data))
        self.labels = numpy.concatenate((self.labels, other.labels))
        return self

######################################## dE_detas DATA IMPORT ########################################

def get_dE_detas_data(data_folder):
    dE_deta_initial = np.loadtxt(f'./{data_folder}/dE_detas_initial')
    dNch_deta_final = np.loadtxt(f'./{data_folder}/dNch_deta_final')

    final_eta = dNch_deta_final[0:1].flatten()

    return final_eta, data_smoothing(dE_deta_initial[1:]), data_smoothing(dNch_deta_final[1:])

class DEData(Data):
    def __init__(self, data_folder):
        self.data_axis, self.data, self.labels = get_dE_detas_data(data_folder)