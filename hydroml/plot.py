import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import Mode
from IPython import display
import numpy as np

def plot_telemetry(d_image_channel, disc_loss_total, outputs, labels):
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(20,15))

    gs = gridspec.GridSpec(4, 1, hspace=0.4)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(disc_loss_total, color='red')
    ax0.set_title("Model Loss")
    ax0.set_ylabel("Mean Squared Error Loss")
    ax0.set_xlabel("Batches")

    ax1 = fig.add_subplot(gs[1,0])
    ax1.plot(outputs[0].flatten(), label="Generated Output", color='red')
    ax1.plot(labels[0].flatten(), '-.', label="Actual Output", color='blue')
    ax1.set_title("Model Output")
    ax1.set_ylabel("$dN^{ch}/d\eta$")
    ax1.set_xlabel("$\eta$")
    ax1.set_ylim([-0.05, 30.05])
    ax1.legend()

    for i in range(len(d_image_channel)):
        image, name, mode = d_image_channel.pop()
        axi = fig.add_subplot(gs[i + 2, 0])
        axi.imshow(image, cmap='plasma', interpolation='bicubic')
        axi.set_title(f'Mode[{mode.name}] - {name}')
        axi.axis('off')

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

        print(total.shape[1])

        assert(total.shape[1] % 2 == 0)

        x_shape = total.shape[0]*2
        y_shape = total.shape[1]//2

        image_channel.push( total.reshape(x_shape, y_shape), layer_name)

        return output
    return hook