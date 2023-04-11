import numpy as np

class Moments:
    """
    Generates the moments data for a given dataset.
    Standard Deviation, Variance, Skew, Kurtosis.

    :param data: The data that the moment will be generated for.
    :type data: numpy.ndarray
    """
    def __init__(self, data):
        self.data = np.array(data)
        self.mean = self.data.mean()
        self.sigma = self.data.std()

        self.second_moment = []
        self.third_moment = []
        self.fourth_moment = []

        for _, data_point in enumerate(data):
            self.second_moment.append( (data_point - self.mean)**2 )
            self.third_moment.append( (data_point - self.mean)**3 )
            self.fourth_moment.append( (data_point - self.mean) ** 4 )

        self.var = np.mean(self.second_moment)
        self.var_error = np.std(self.second_moment/np.sqrt(len(data)))

        self.skew = np.mean(self.third_moment)/self.sigma**3
        self.skew_error = np.std(self.third_moment)/np.sqrt(len(data) * self.sigma**3)

        self.kurt = np.mean(self.fourth_moment)/self.sigma**4
        self.kurt_error = np.std(self.fourth_moment)/np.sqrt(len(data) * self.sigma**4)

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