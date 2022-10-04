import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# def plot_telemetry(outputs, labels, etas):
#     fig = plt.figure(figsize=(20,15))
#
#     ax1 = fig.add_subplot()
#     ax1.plot(etas, outputs[0].flatten(), label="Generated Output", color='red')
#     ax1.plot(etas, labels[0].flatten(), '-.', label="Actual Output", color='blue')
#     ax1.set_title("Model Output")
#     ax1.set_ylabel("$dN^{ch}/d\eta$")
#     ax1.set_xlabel("$\eta$")
#     ax1.legend()
#
#     plt.show()

def plot_output(eta, model_output, actual_output):
    """
    Plots the output of the provided output/input pair.

    :param eta: The x-axis (eta)
    :type eta: numpy.array

    :param model_output: The output of the model
    :type model_output: numpy.array

    :param actual_output: The final state distribution.
    :type actual_output: numpy.array
    """
    fig = plt.figure(figsize=(15,5))

    ax = fig.add_subplot()
    ax.plot(eta, model_output.flatten(), label="Generated Pseudorapidity Distribution", color='red')
    ax.plot(eta, actual_output.flatten(), '-.', label="Real Pseudorapidity Distribution", color='blue')
    ax.set_title("Model Output")
    ax.set_ylabel("$dN^{ch}/d\eta$")
    ax.set_xlabel("$\eta$")
    ax.legend()

    plt.show()

def plot_cc_graph(actual, generated, type):
    """
    Plots the correlation graph of the integrated curves.

    :param actual: The final state distribution.
    :type actual: numpy.array

    :param generated: The model output of the final state distribution.
    :type generated: numpy.array

    :param type: The LaTeX string for the output of the integration.
    :type type: string
    """
    fig = plt.figure(figsize=(10, 10))

    ax0 = fig.add_subplot()

    xy = np.vstack((actual, generated))

    z1 = stats.gaussian_kde(xy)(xy)

    color_map = 'gnuplot2'

    ax0.scatter(actual, generated, c=z1, cmap=color_map)

    ax0.set_xlabel(f'${type}$(Actual)')
    ax0.set_ylabel(f'${type}$(Generated)')
    ax0.set_title(f'Actual vs Generated ${type}$')

    box_text = f'r = {stats.pearsonr(actual, generated)[0]:.4f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax0.text(0.05, 0.95, box_text, verticalalignment='top',
             bbox=props, transform=ax0.transAxes)

    ax0.plot([0, actual.max()], [0, generated.max()], color='red')

    plt.show()