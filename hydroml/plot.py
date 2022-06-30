import matplotlib.pyplot as plt
from IPython import display

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
    ax[1].legend()

    plt.show()
