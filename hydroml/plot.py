import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import display
import numpy as np
import numpy.linalg

def plot_telemetry(disc_loss_total, gen_loss_total, real_total, fake_total, data, generated_data):
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(30,10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4)

    ax0 = fig.add_subplot(gs[0, 0 ])
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

        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.imshow(total.reshape(x_shape, y_shape), cmap='plasma', interpolation='bicubic')
        ax.set_title('Learned Feature for Selected Layer')
        ax.axis('off')
        plt.show()

        return output
    return hook