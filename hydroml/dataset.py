import sys
import os
import numpy as np
from torch.utils.data import Dataset as DS
from scipy.interpolate import interp1d

class Dataset(DS):
    def __init__(self, dataset, size=sys.maxsize):
        self.initial = np.empty()
        self.final = np.empty()
        self.start_eta = np.empty()
        self.final_eta = np.empty()

    def __len__(self):
        return len(self.initial)

    def __getitem__(self, item):
        return self.initial[item], self.final[item]

    # Using multiple datasets, add them together
    def __add__(self, other):
        self.initial = np.concatenate((self.initial, other.data))
        self.final = np.concatenate((self.final, other.labels))
        return self

    def delete_elements(self, to_remove):
        self.initial = np.delete(self.initial, to_remove, 0)
        self.final = np.delete(self.final, to_remove, 0)

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

        for _, data in enumerate(self.initial):
            new_data.append(data[indices[0] : indices[-1] + 1])

        for _, label in enumerate(self.final):
            new_labels.append(label[indices[0] : indices[-1] + 1])

        self.start_eta = np.array(sum_x_axis)
        self.final_eta = np.array(sum_x_axis)
        self.initial = np.array(new_data, dtype=np.float64)
        self.final = np.array(new_labels, dtype=np.float64)
        return self

    def interpolate(self, resolution=200):
        new_data = []
        new_labels = []

        x_new_start = np.linspace(self.start_eta[0], self.start_eta[-1], num=resolution)
        x_new_final = np.linspace(self.final_eta[0], self.final_eta[-1], num=resolution)

        for i, data in enumerate(self.initial):
            d = interp1d( self.start_eta, data )
            l = interp1d(self.final_eta, self.final[i])

            new_data.append(d( x_new_start ))
            new_labels.append( l( x_new_final ) )

        new_eta_start = np.linspace(self.start_eta[0], self.start_eta[-1], num=resolution)
        new_eta_final = np.linspace(self.final_eta[0], self.final_eta[-1], num=resolution)

        self.initial = np.array(new_data, dtype=np.float64)
        self.final = np.array(new_labels, dtype=np.float64)
        self.final_eta = np.array(new_eta_final)
        self.start_eta = np.array(new_eta_start)
        return self

    # Returning a null dataset. Needs to be fixed.
    def smooth(self):
        for i, data in enumerate(self.initial):
            for j in range(2, len(data) - 2):
                average = np.float64((data[j - 2] + data[j - 1] + data[j] + data[j + 1] + data[j + 2]) / 5)
                data[j] = average
            self.initial[i] = data

class BaryonDataset(Dataset):
    def __init__(self, dataset, size=sys.maxsize):
        assert(size >= 1)

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

            # Low energy filter
            if proton.max() < 5.:
                continue

            baryons.append( baryon.reshape(1, 141) )
            protons.append( proton.reshape(1, 141) )

        self.data = np.array( baryons, dtype=np.float64 )
        self.labels = np.array( protons, dtype=np.float64 )
        self.start_eta = np.array( eta_baryon, dtype=np.float64 )
        self.final_eta = np.array( eta_proton, dtype=np.float64 )

class EnergyDensityDataset(Dataset):
    def __init__(self, data_folder, standardize=False):
        dE_deta_initial = np.loadtxt(f'./{data_folder}/dE_detas_initial')
        dNch_deta_final = np.loadtxt(f'./{data_folder}/dET_deta_final')

        self.start_eta = dE_deta_initial[0:1].flatten()
        self.final_eta = dNch_deta_final[0:1].flatten()

        self.data = np.array( dE_deta_initial[1:], dtype=np.float64 )
        self.labels = np.array( dNch_deta_final[1:], dtype=np.float64 )

        if standardize:
            self.data = ((dE_deta_initial - np.mean(dE_deta_initial, axis=0)) / (
                        np.std(dE_deta_initial, axis=0) + 1e-16))
            self.labels = ((dNch_deta_final - np.mean(dNch_deta_final, axis=0)) / (
                        np.std(dNch_deta_final, axis=0) + 1e-16))