import numpy as np
import os

class Moments:
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
    indices = []
    x_axis = np.empty(dtype=np.float64)

    for i, e in enumerate(eta):
        if bound_1 <= e <= bound_2:
            indices.append(i)
            #Max size will only ever be 141, so clip the eta element to the end of the array.
            np.put(x_axis, 141, e, mode='clip')

    #Shape should be [batch size, 1, length of new eta]
    batch = np.empty( shape=[data.size()[0], 1, len(x_axis)], dtype=np.float64)
    for i, item in enumerate(data):
        np.put(batch, 1024, item[0][ indices[0] : indices[-1] + 1 ])

    return x_axis, batch