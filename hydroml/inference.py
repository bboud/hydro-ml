import numpy as np
import torch

def inference(model, data):
    """
    Takes a single initial state array of size [462] between eta values of -4.9, -3.1.

    If the data being fed into this model is not the correct size, the model will not be able to perform the inference.
    The model is trained on a specific dataset size that cannot be changed after the fact.

    If your data is not already trimmed down to the specified range, you can call hydroml.utils.trim(eta, data, bound_1, bound_2).

    You must load the model using pytorch! This function is basically a wrapper to verify the data before being passed to the model.

    :param model: The loaded DEConvolution Model
    :type model: model.DEConvolutionModel

    :param data: The initial state distribution.
    :type data: numpy.ndarray

    :return: The predicted final state psudorapidity distribution.
    :rtype: numpy.ndarray
    """

    #Flatten in case there are some weird shapes such as [[462],[]] or something similar.
    if(type(data) is not np.ndarray):
        data = np.array(data).flatten()
    else:
        data = data.flatten()

    assert len(data) == 462, f"The model takes in an array of size 462, your array is size {len(data)}"

    return model(torch.Tensor(data)).detach().numpy()

def smooth(data):
    """
    Smooths the data by taking the running average in order to remove noise.

    :param data: The data to be smoothed.
    :type data: numpy.ndarray

    :return: Data that has been smoothed.
    :rtype: numpy.ndarray
    """

    for j in range(2, len(data) - 2):
        average = np.float64((data[j - 2] + data[j - 1] + data[j] + data[j + 1] + data[j + 2]) / 5)
        data[j] = average

    return data