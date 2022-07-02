import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from hydroml.utils import Mode, ModelType
from IPython import display
import numpy as np

def plot_telemetry(d_image_channel, g_image_channel, disc_loss_total, gen_loss_total, real_total, fake_total, data, generated_data):
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(30,30))

    gs = gridspec.GridSpec(12, 2, hspace=0.6)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(disc_loss_total, label = 'Discriminator Loss', color='red')
    ax0.plot(gen_loss_total, label='Generator Loss', color='blue')
    ax0.set_title("Model Loss")
    ax0.set_ylabel("Binary Cross Entropy Loss")
    ax0.set_xlabel("Batches")
    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 1 ])
    ax1.plot(real_total, label="Mean Real Output", color='blue')
    ax1.plot(fake_total, label="Mean Fake Output", color='red')
    ax1.set_title("Mean Output")
    ax1.set_ylabel("Mean")
    ax1.set_xlabel("Batches")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend()

    ax2  = fig.add_subplot(gs[1, 0])
    ax2.plot(generated_data[0][0], label="Generated Curve", color='red')
    ax2.set_title("Generator Output")
    ax2.set_ylabel("$dN^{ch}/d\eta$")
    ax2.set_xlabel("$\eta$")
    ax2.set_ylim([-0.05, 1.05])

    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(generated_data[1][0], label="Mean Fake Output", color='red')
    ax3.set_title("Generator Output")
    ax3.set_ylabel("$dN^{ch}/d\eta$")
    ax3.set_xlabel("$\eta$")
    ax3.set_ylim([-0.05, 1.05])

    for i in range(len(d_image_channel)):
        image, name, mode = d_image_channel.pop()
        axi = fig.add_subplot(gs[i + 2, 0])
        axi.imshow(image, cmap='plasma', interpolation='bicubic')
        axi.set_title(f'Type[DISCRIMINATOR] - Mode[{mode.name}] - {name}')
        axi.axis('off')

    for i in range(len(g_image_channel)):
        image, name, mode= g_image_channel.pop()
        axi = fig.add_subplot(gs[i + 2, 1])
        axi.imshow(image, cmap='plasma', interpolation='bicubic')
        axi.set_title(f'Type[GENERATOR] - Mode[{mode.name}] - {name}')
        axi.axis('off')

    plt.show()

def discriminator_layer_hook(layer_name, image_channel):
    def hook(module, input, output):
        if image_channel.get_mode() == Mode.NOISE:
            return

        output_array = output.detach().numpy()

        total = []

        for i in range(output_array.shape[0]):
            total.append(output_array[i].flatten())

        total = np.array(total)

        assert(total.shape[1] % 2 == 0)

        x_shape = total.shape[0]*2
        y_shape = total.shape[1]//2

        image_channel.push( total.reshape(x_shape, y_shape), layer_name)

        return output
    return hook

def generator_layer_hook(layer_name, image_channel):
    def hook(module, input, output):
        output_array = output.detach().numpy()

        total = []

        for i in range(output_array.shape[0]):
            total.append(output_array[i].flatten())

        total = np.array(total)

        assert(total.shape[1] % 2 == 0)

        x_shape = total.shape[0]*2
        y_shape = total.shape[1]//2

        image_channel.push( total.reshape(x_shape, y_shape), layer_name)

        return output
    return hook