import sys
sys.path.append('../hydroml')
from hydroml.utils import trim

import os
import numpy as np
from torch.utils.data import Dataset as DS
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod

"""
    An abstract class to represent basic operations on different datasets.
    
    ...

    Attributes
    ----------
    initial : numpy.array
        Initial state distribution
    final : numpy.array
        Final state distribution
    start_eta : numpy.array
        X-Axis that corresponds to the initial state distribution.
    final_eta : numpy.array
        X-Axis that corresponds to the final state distribution.

    Methods
    -------
    delete_elements(to_remove):
        Removes data from both the initial and final state distributions using a single or array of indices.
    trim(bound_1, bound_1):
        Trims the initial and final state distributions down to a specific range of eta.
    interpolate(resolution=200):
        Will interpolate the initial and final state distributions to a specific resolution. 
    smooth():
        Smooths the data by taking the running average over the initial and final state distributions in order to remove noise.
    standardize():
        Standardises the initial and final state distributions.
    """
class Dataset(DS, ABC):
    @abstractmethod
    def __init__(self):
        """
        Constructor must be overridden by a specific dataset class.
        """
        pass

    # Will return the length of all member elements of the dataset.
    def __len__(self):
        return len(self.initial)

    def __getitem__(self, item):
        return self.initial[item], self.final[item]

    # Using multiple datasets, add them together. They must have the same eta ranges.
    def __add__(self, other):
        assert len(other.start_eta) == len(self.start_eta) and len(other.final_eta) == len(self.final_eta)
        self.initial = np.concatenate((self.initial, other.initial))
        self.final = np.concatenate((self.final, other.final))
        return self

    def delete_elements(self, to_remove):
        """
        Removes data from both the initial and final state distributions using a single or array of indices.
        :param to_remove: int, numpy.array
            The integer index or an array of indices that specify which elements to remove.
        :return: dataset
        """
        self.initial = np.delete(self.initial, to_remove, 0)
        self.final = np.delete(self.final, to_remove, 0)

        return self

    #Trim the whole dataset down to a specific range.
    def trim(self, bound_1, bound_2):
        """
        Trims the initial and final state distributions down to a specific range of eta.
        :param bound_1: float
            The left most eta bound.
        :param bound_2: float
            The right most eta bound.
        :return: dataset
        """
        indices_start = []
        indices_final = []
        sum_x_axis_start = []
        sum_x_axis_final = []
        new_data = []
        new_labels = []

        for i, eta in enumerate(self.start_eta):
            if bound_1 <= eta <= bound_2:
                indices_start.append(i)
                sum_x_axis_start.append(eta)

        for i, eta in enumerate(self.final_eta):
            if bound_1 <= eta <= bound_2:
                indices_final.append(i)
                sum_x_axis_final.append(eta)

        for _, data in enumerate(self.initial):
            new_data.append(data[indices_start[0] : indices_start[-1] + 1])

        for _, label in enumerate(self.final):
            new_labels.append(label[indices_final[0] : indices_final[-1] + 1])

        self.start_eta = np.array(sum_x_axis_start)
        self.final_eta = np.array(sum_x_axis_final)
        self.initial = np.array(new_data, dtype=np.float64)
        self.final = np.array(new_labels, dtype=np.float64)
        return self

    # Interpolate the whole dataset to a desired resolution.
    def interpolate(self, resolution=200):
        """
        Will interpolate the initial and final state distributions to a specific resolution.
        :param resolution: int
            The number of datapoints that are desired.
        :return: dataset
        """
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

    # Smooth the whole dataset.
    def smooth(self):
        """
        Smooths the data by taking the running average over the initial and final state distributions in order to remove noise.
        :return: dataset
        """
        for i, data in enumerate(self.initial):
            for j in range(2, len(data) - 2):
                average = np.float64((data[j - 2] + data[j - 1] + data[j] + data[j + 1] + data[j + 2]) / 5)
                data[j] = average
            self.initial[i] = data
        for i, data in enumerate(self.final):
            for j in range(2, len(data) - 2):
                average = np.float64((data[j - 2] + data[j - 1] + data[j] + data[j + 1] + data[j + 2]) / 5)
                data[j] = average
            self.final[i] = data
        return self

    def standardize(self):
        """
        Standardises the initial and final state distributions.
        :return: dataset
        """
        self.initial = ((self.initial - np.mean(self.initial, axis=0)) / (
                np.std(self.initial, axis=0) + 1e-16))
        self.final = ((self.final - np.mean(self.final, axis=0)) / (
                np.std(self.final, axis=0) + 1e-16))
        return self

class BaryonDataset(Dataset):
    def __init__(self, dataset, size=sys.maxsize):
        assert(size >= 1)

        events = []
        i = 0
        for file in os.listdir(dataset):
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
                f'{dataset}/event_{event}_net_baryon_etas.txt', unpack=True)
            eta_proton, proton, error = np.loadtxt(
                f'{dataset}/event_{event}_net_proton_eta.txt', unpack=True)

            # Low energy filter
            if proton.max() < 5.:
                continue

            baryons.append( baryon.reshape(1, 141) )
            protons.append( proton.reshape(1, 141) )

        self.initial = np.array( baryons, dtype=np.float64 )
        self.final = np.array( protons, dtype=np.float64 )
        self.start_eta = np.array( eta_baryon, dtype=np.float64 )
        self.final_eta = np.array( eta_proton, dtype=np.float64 )


"""
    Dataset class that is responsible for data related to the energy density model.

    ...

    Attributes
    ----------
    initial : numpy.array
        Initial state distribution
    final : numpy.array
        Final state distribution
    start_eta : numpy.array
        X-Axis that corresponds to the initial state distribution.
    final_eta : numpy.array
        X-Axis that corresponds to the final state distribution.

    Methods
    -------
    delete_elements(to_remove):
        Removes data from both the initial and final state distributions using a single or array of indices.
    trim(bound_1, bound_1):
        Trims the initial and final state distributions down to a specific range of eta.
    interpolate(resolution=200):
        Will interpolate the initial and final state distributions to a specific resolution. 
    smooth():
        Smooths the data by taking the running average over the initial and final state distributions in order to remove noise.
    standardize():
        Standardises the initial and final state distributions.
    cosh():
        Will divide the initial state distribution by the hyperbolic cosine of the x-axis (eta).
    no_asymmetric():
        Will remove the asymmetric datapoints from the dataset. Not recommended for use. Please see remove_anomalies.
    remove_anomalies(threshold=150):
        Will remove data that falls out of a specific energy threshold.    
    """
class EnergyDensityDataset(Dataset):
    def __init__(self, initial_file, final_file):
        dE_deta_initial = np.loadtxt(initial_file)
        dNch_deta_final = np.loadtxt(final_file)

        self.start_eta = dE_deta_initial[0:1].flatten()
        self.final_eta = dNch_deta_final[0:1].flatten()

        self.initial = np.array( dE_deta_initial[1:], dtype=np.float64 )
        self.final = np.array( dNch_deta_final[1:], dtype=np.float64 )

    def cosh(self):
        """
        Will divide the initial state distribution by the hyperbolic cosine of the x-axis (eta).
        :return: dataset
        """
        self.initial = self.initial/np.cosh(self.start_eta)
        return self

    def no_asymmetric(self):
        """
        Will remove the asymmetric datapoints from the dataset. Not recommended for use. Please see remove_anomalies.
        :return: dataset
        """
        to_remove = []
        for i, curve in enumerate(self.final):
            trim_axis_left, trim_curve_left = trim(self.final_eta, curve, self.final_eta[0], 0)
            left_integral = np.trapz(trim_curve_left, trim_axis_left)

            trim_axis_right, trim_curve_right = trim(self.final_eta, curve, 0, self.final_eta[-1])
            right_integral = np.trapz(trim_curve_right, trim_axis_right)

            difference = left_integral - right_integral
            if difference > 200 or difference < -200:
                to_remove.append(i)

        return self.delete_elements(to_remove)

    def remove_anomalies(self, threshold):
        """
        Will remove data that falls out of a specific energy threshold.
        :param threshold:
        :return: dataset
        """
        to_remove = []

        for i, data in enumerate(self):
            # Trim down the data to the section that we want.
            trim_final_axis, trim_final = trim(self.final_eta, data[1], -4.9, -3.1)

            integrated_final = np.trapz(trim_final, trim_final_axis)

            if integrated_final > threshold:
                print(f'Removing events {i}')
                to_remove.append(i)

        return self.delete_elements(to_remove)
