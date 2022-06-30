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

        # There are a large number of these, but we will only sample 4.
        fig = plt.figure(figsize=(20,5))
        ax = fig.subplots( 4, 4 )
        for i in range(4):
            for j in range(4):
                ax[i][j].plot( np.arange( len(output_array[i][j]) ), output_array[i][j])
        plt.show()

        return output
    return hook