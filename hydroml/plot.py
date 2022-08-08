import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

def plot_telemetry( loss_total, outputs, labels, etas=None):
    fig = plt.figure(figsize=(20,15))

    gs = gridspec.GridSpec(2, 1)

    # ax0 = fig.add_subplot(gs[0, : ])
    # ax0.plot(loss_total, color='red')
    # ax0.set_title("Model Loss")
    # ax0.set_ylabel("Mean Squared Error Loss")
    # ax0.set_xlabel("Batches")
    # #ax0.set_ylim([-0.05, 5.05])

    if etas is not None:
        x = etas
    else:
        x = np.linspace(-6.4, 6.4, 141)

    ax1 = fig.add_subplot(gs[1, : ])
    ax1.plot(x, outputs[0].flatten(), label="Generated Output", color='red')
    ax1.plot(x, labels[0].flatten(), '-.', label="Actual Output", color='blue')
    ax1.set_title("Model Output")
    ax1.set_ylabel("$dN^{ch}/d\eta$")
    ax1.set_xlabel("$\eta$")
    #ax1.set_ylim([-0.05, 30.05])
    ax1.legend()

    plt.show()

def plot_output(model_output, actual_output, eta):
    fig = plt.figure(figsize=(15,5))

    ax = fig.add_subplot()
    ax.plot(eta, model_output.flatten(), label="Generated Output", color='red')
    ax.plot(eta, actual_output.flatten(), '-.', label="Actual Output", color='blue')
    ax.set_title("Model Output")
    ax.set_ylabel("$dN^{ch}/d\eta$")
    ax.set_xlabel("$\eta$")
    ax.legend()

    plt.show()

def plot_cc_graph(actual, generated):
    fig = plt.figure(figsize=(10, 10))

    ax0 = fig.add_subplot()

    xy = np.vstack((actual, generated))

    z1 = stats.gaussian_kde(xy)(xy)

    color_map = 'gnuplot2'

    ax0.scatter(actual, generated, c=z1, cmap=color_map)

    ax0.set_xlabel('$N^{ch}$(Actual)')
    ax0.set_ylabel('$N^{ch}$(Generated)')
    ax0.set_title('Actual vs Generated $N^{ch}$')

    box_text = f'r = {stats.pearsonr(actual, generated)[0]:.4f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax0.text(0.05, 0.95, box_text, verticalalignment='top',
             bbox=props, transform=ax0.transAxes)

    plt.show()