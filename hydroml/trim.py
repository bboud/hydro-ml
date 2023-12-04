import numpy as np

#Trim batch or single data outside of the whole dataset
def trim(eta, data, bound_1, bound_2):
    """
    Trims a single data point into specific range. If you need to trim a batch, please see 'batch_trim'.
    :param eta: The x-axis (eta)
    :type eta: numpy.ndarray
    :param data: The data will be trimmed.
    :type data: numpy.ndarray
    :param bound_1: The left most eta bound.
    :type bound_1: float
    :param bound_2: The right most eta bound.
    :type bound_2: float
    :return: Returns the datapoint that has been trimmed.
    :rtype: numpy.ndarray
    """
    indices = []
    x_axis = []

    for i, e in enumerate(eta):
        if bound_1 <= e <= bound_2:
            indices.append(i)
            x_axis.append(e)

    return np.array(x_axis, dtype=np.float32), np.array(data[ indices[0] : indices[-1] + 1 ], dtype=np.float32)

def batch_trim(eta, batch, bound_1, bound_2):
    """
    Trims a batch of data points into specific range.
    :param eta: The x-axis (eta)
    :type eta: numpy.ndarray

    :param data: The batch of data that will be trimmed.
    :type data: numpy.ndarray

    :param bound_1: The left most eta bound.
    :type bound_1: float

    :param bound_2: The right most eta bound.
    :type bound_2: float

    :return: Returns the batch of datapoints that has been trimmed.
    :rtype: numpy.ndarray
    """
    output_eta = None
    output_data = []
    for data in batch:
        trim_eta, trim_data = trim(eta, data.flatten(), bound_1, bound_2)
        if output_eta is None:
            output_eta = trim_eta
        output_data.append( trim_data )

    output_data = np.array(output_data, dtype=np.float32)

    return np.array( output_eta, dtype=np.float32 ), np.array(output_data, dtype=np.float32)