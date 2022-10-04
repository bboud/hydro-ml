import numpy as np
from torch import tensor

class Moments:
    """
    Generates the moments data for a given dataset.
    Standard Deviation, Variance, Skew, Kurtosis.

    :param data: The data that the moment will be generated for.
    :type data: numpy.array
    """
    def __init__(self, data):
        self.data = np.array(data)
        self.mean = self.data.mean()
        self.sigma = self.data.std()

        second_moment = []
        third_moment = []
        fourth_moment = []

        for _, data_point in enumerate(data):
            second_moment.append( (data_point - self.mean)**2 )
            third_moment.append( (data_point - self.mean)**3 )
            fourth_moment.append( (data_point - self.mean) ** 4 )

        self.var = np.mean(second_moment)
        self.var_error = np.std(second_moment/np.sqrt(len(data)))

        self.skew = np.mean(third_moment)/self.sigma**3
        self.skew_error = np.std(third_moment)/np.sqrt(len(data) * self.sigma**3)

        self.kurt = np.mean(fourth_moment)/self.sigma**4
        self.kurt_error = np.std(fourth_moment)/np.sqrt(len(data) * self.sigma**4)

    def __str__(self):
        formatted_string = f'Mean: {self.mean}\n' \
                           f'Standard Diviation: {self.sigma}\n' \
                           f'Variance: {self.var} \n' \
                           f'Variance Error: {self.var_error}\n' \
                           f'Skew: {self.skew}\n' \
                           f'Skew Error: {self.skew_error}\n' \
                           f'Kurtosis: {self.kurt}\n' \
                           f'Kurtosis Error: {self.kurt_error}'

        return formatted_string

#Trim batch or single data outside of the whole dataset
def trim(eta, data, bound_1, bound_2):
    """
    Trims a single data point into specific range. If you need to trim a batch, please see 'batch_trim'.
    :param eta: The x-axis (eta)
    :type eta: numpy.array

    :param data: The data will be trimmed.
    :type data: numpy.array

    :param bound_1: The left most eta bound.
    :type bound_1: float

    :param bound_2: The right most eta bound.
    :type bound_2: float

    :return: Returns the datapoint that has been trimmed.
    :rtype: numpy.array
    """
    indices = []
    x_axis = []

    for i, e in enumerate(eta):
        if bound_1 <= e <= bound_2:
            indices.append(i)
            x_axis.append(e)

    return np.array(x_axis, dtype=np.float64), np.array(data[ indices[0] : indices[-1] + 1 ], dtype=np.float64)

def batch_trim(eta, batch, bound_1, bound_2):
    """
    Trims a batch of data points into specific range.
    :param eta: The x-axis (eta)
    :type eta: numpy.array

    :param data: The batch of data that will be trimmed.
    :type data: numpy.array

    :param bound_1: The left most eta bound.
    :type bound_1: float

    :param bound_2: The right most eta bound.
    :type bound_2: float

    :return: Returns the batch of datapoints that has been trimmed.
    :rtype: numpy.array
    """
    output_eta = None
    output_data = []
    for data in batch:
        trim_eta, trim_data = trim(eta, data.flatten(), bound_1, bound_2)
        if output_eta is None:
            output_eta = trim_eta
        output_data.append( trim_data )

    # Calling tensor directly is 'slow'
    output_data = tensor(np.array(output_data)).reshape(batch.shape[0], 1, len(output_eta))

    return np.array( output_eta, dtype=np.float64 ), output_data