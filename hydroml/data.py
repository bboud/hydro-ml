import numpy
import numpy as np
import os
import sys
from torch.utils.data import Dataset

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
        if proton.max() < 5.:
            continue

        baryon = data_smoothing(baryon)
        proton = data_smoothing(proton)

        baryons.append(baryon.reshape(1, 141))
        protons.append(proton.reshape(1, 141))

    return np.array(baryons, dtype=np.float64), np.array(protons, dtype=np.float64), np.array(eta_baryon, dtype=np.float64), np.array(eta_proton, dtype=np.float64)

def data_smoothing(data):
    # Calculate the moving average
    for i in range( 2, len(data) - 2 ):
        average = np.float64( (data[i-2]  + data[i-1] + data[i] + data[i+1] + data[i+2]) / 5 )
        data[i] = average
    return data

class Data(Dataset):
    def __init__(self, dataset, size=sys.maxsize):
        assert(size >= 1)
        self.data, self.labels, self.start_eta, self.final_eta = get_real_data(dataset, size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    # Using multiple datasets, add them together
    def __add__(self, other):
        self.data = numpy.concatenate((self.data, other.data))
        self.labels = numpy.concatenate((self.labels, other.labels))
        return self

    def trim(self, bound_1, bound_2):
        indices = []
        sum_x_axis = []
        new_data = []
        new_labels = []

        for i, eta in enumerate(self.final_eta):
            if bound_1 <= eta <= bound_2:
                indices.append(i)
                sum_x_axis.append(eta)

        for _, data in enumerate(self.data):
            new_data.append(data[indices[0] : indices[-1] + 1])

        for _, label in enumerate(self.labels):
            new_labels.append(label[indices[0] : indices[-1] + 1])

        self.start_eta = np.array(sum_x_axis)
        self.final_eta = np.array(sum_x_axis)
        self.data = np.array(new_data)
        self.labels = np.array(new_labels)
        return self

######################################## dE_detas DATA IMPORT ########################################

def get_dE_detas_data(data_folder):
    dE_deta_initial = np.loadtxt(f'./{data_folder}/dE_detas_initial')
    dNch_deta_final = np.loadtxt(f'./{data_folder}/dNch_deta_final')

    start_eta = dE_deta_initial[0:1].flatten()
    final_eta = dNch_deta_final[0:1].flatten()

    return start_eta, final_eta, data_smoothing(dE_deta_initial[1:]), data_smoothing(dNch_deta_final[1:])

class DEData(Data):
    def __init__(self, data_folder):
        self.start_eta, self.final_eta, self.data, self.labels = get_dE_detas_data(data_folder)