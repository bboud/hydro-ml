import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_telemetry( loss_total, outputs, labels, etas=None):
    fig = plt.figure(figsize=(20,15))

    gs = gridspec.GridSpec(2, 1)

    ax0 = fig.add_subplot(gs[0, : ])
    ax0.plot(loss_total, color='red')
    ax0.set_title("Model Loss")
    ax0.set_ylabel("Mean Squared Error Loss")
    ax0.set_xlabel("Batches")
    #ax0.set_ylim([-0.05, 5.05])

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

def plot_output(outputs, labels):
    fig = plt.figure(figsize=(15,5))

    gs = gridspec.GridSpec(len(outputs), 1, hspace=len(outputs)*0.1)

    x = np.linspace(-6.4, 6.4, 141)

    for i in range(len(outputs)):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(x, outputs[i].flatten(), label="Generated Output", color='red')
        ax.plot(x, labels[i].flatten(), '-.', label="Actual Output", color='blue')
        ax.set_title("Model Output")
        ax.set_ylabel("$dN^{ch}/d\eta$")
        ax.set_xlabel("$\eta$")
        ax.legend()

    plt.show()