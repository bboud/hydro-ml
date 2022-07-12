import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import Mode
from IPython import display
import numpy as np

def plot_telemetry(d_image_channel, disc_loss_total, outputs, labels, etas=None):
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(20,15))

    gs = gridspec.GridSpec(3, len(d_image_channel))

    ax0 = fig.add_subplot(gs[0, : ])
    ax0.plot(disc_loss_total, color='red')
    ax0.set_title("Model Loss")
    ax0.set_ylabel("Mean Squared Error Loss")
    ax0.set_xlabel("Batches")
    ax0.set_ylim([-0.05, 5.05])

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

    for i in range(len(d_image_channel)):
        image, name, mode = d_image_channel.pop()
        axi = fig.add_subplot(gs[2, i])
        axi.imshow(image, cmap='plasma', interpolation='bicubic')
        axi.set_title(f'Mode[{mode.name}] - {name}')
        axi.axis('off')

    plt.show()

def plot_output(outputs, labels):
    fig = plt.figure(figsize=(10,35))

    gs = gridspec.GridSpec(len(outputs), 1, hspace=len(outputs)*0.1)

    print(outputs.shape)

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


def simple_layer_hook(layer_name, image_channel):
    def hook(module, input, output):
        if image_channel.get_mode() == Mode.NOISE:
            return

        output_array = output.detach().numpy()

        total = []

        for i in range(output_array.shape[0]):
            total.append(output_array[i].flatten())

        total = np.array(total)

        assert(total.shape[1] % 3 == 0)

        x_shape = total.shape[0]*3
        y_shape = total.shape[1]//3

        image_channel.push( total.reshape(x_shape, y_shape), layer_name)

        return output
    return hook