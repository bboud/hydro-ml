from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
from IPython import display
import numpy as np

def plot_telemetry(loss_total, real_total, fake_total):
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(15,10))
    ax = fig.subplots(2,1)

    ax[0].plot(loss_total, '-o')
    ax[0].set_title("Discriminator Loss")
    ax[0].set_ylabel("Binary Cross Entropy Loss")
    ax[0].set_xlabel("Iterations")

    ax[1].plot(real_total, '-o', label="Mean Real Output")

    ax[1].plot(fake_total, '-o', label="Mean Fake Output")

    ax[1].set_title("Mean Output")
    ax[1].set_ylabel("Mean")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylim([-0.05, 1.05])
    ax[1].legend()

    plt.show()

def layer_hook():
    def hook(module, input, output):
        output_array = output.detach().numpy()

        total = []

        for i in range(output_array.shape[0]):
            total.append(output_array[i].flatten())

        total = np.array(total)

        assert(total.shape[1] % 2 == 0)

        x_shape = total.shape[0]*2
        y_shape = total.shape[1]//2

        half_shape = (total.shape[0]*total.shape[1]) // 2

        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.imshow(total.reshape(x_shape, y_shape), cmap='tab20b', interpolation='bicubic')
        ax.set_title('Learned Feature for Selected Layer')
        ax.axis('off')
        plt.show()

        return output
    return hook