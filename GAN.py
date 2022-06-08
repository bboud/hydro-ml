from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def Kernel(x, x0):
    sigma = 0.8
    protonFraction = 0.4
    norm = protonFraction/(np.sqrt(2.*np.pi)*sigma)
    return(norm*np.exp(-(x - x0)**2./(2.*sigma**2.)))

def FakeKernel(x, x0):
    sigma = 0.4 #Just a slight nudge
    protonFraction = 0.4
    norm = protonFraction/(np.sqrt(2.*np.pi)*sigma)
    return(norm*np.exp(-(x - x0)**2./(2.*sigma**2.)))


def test_data_gen(fakeKernel=False):
    A = 197
    yBeam = 5.36
    slope = 0.5
    sigmaEtas = 0.2

    # generate input data
    nBaryons = np.random.randint(0, 2 * A)
    randX = np.random.uniform(0, 1, size=nBaryons)
    etasBaryon = 1. / slope * np.arcsinh((2. * randX - 1) * np.sinh(slope * yBeam))
    etasArr = np.linspace(-6.4, 6.4, 128)
    dNBdetas = np.zeros(len(etasArr))
    norm = 1. / (np.sqrt(2. * np.pi) * sigmaEtas)
    for iB in etasBaryon:
        dNBdetas += norm * np.exp(-(etasArr - iB) ** 2. / (2. * sigmaEtas ** 2.))

    # generate test data with convolution with a kernel
    dNpdy = np.zeros(len(etasArr))
    detas = etasArr[1] - etasArr[0]
    for i in range(len(etasArr)):
        dNpdy[i] = sum(Kernel(etasArr, etasArr[i]) * dNBdetas) * detas

    if fakeKernel:
        # generate test data with convolution with a fake kernel
        dNBdetasFake = np.random.uniform(0.0, dNBdetas.max(), size=len(etasArr))
        dNpdyFake = np.zeros(len(etasArr))
        detas = etasArr[1] - etasArr[0]
        for i in range(len(etasArr)):
            dNpdyFake[i] = sum(FakeKernel(etasArr, etasArr[i]) * dNBdetas) * detas

        return (etasArr, dNBdetas, dNpdy, dNBdetas, dNpdyFake)
    else:
        # generate fake data with random noise
        dNBdetasFake = np.random.uniform(0.0, dNBdetas.max(), size=len(etasArr))
        dNpdyFake = np.random.uniform(0.0, dNpdy.max(), size=len(etasArr))

        return (etasArr, dNBdetas, dNpdy, dNBdetasFake, dNpdyFake)


def generate_data(size=500):
    dataArr = []
    labelArr = []
    for iev in range(size):
        x, y1, y2, y3, y4 = test_data_gen(fakeKernel=False)

        # real data
        x = y2
        dataArr.append(x)
        labelArr.append(1)

        # fake data: random
        x = y4
        dataArr.append(x)
        labelArr.append(0)

        x, y1, y2, y3, y4 = test_data_gen(fakeKernel=True)

        # real data
        x = y2
        dataArr.append(x)
        labelArr.append(1)

        # fake data: FakeKernel
        x = y4
        dataArr.append(x)
        labelArr.append(0)

    return (np.array(dataArr), np.array(labelArr))

def define_discriminator(dimShape=(128,)):
    #Sequential model
    model = Sequential([
        #Acurracy is achieved much faster with 4 input nodes.
        layers.Dense(units=4, activation='relu', input_shape=dimShape),
        layers.Dense(units=1, activation='sigmoid')
    ])
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    return(model)


def define_generator():
    model = Sequential([
        # First set of layers will be low resolution noise.
        layers.Dense(32 * 200, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Now we will reshape and upsample to get the final output dimension which should be (128,)
        layers.Reshape((32, 200)),

        layers.Conv1DTranspose(64, 1, strides=2),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv1DTranspose(128, 1, strides=2),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv1DTranspose(1, 1, activation="tanh")
    ])
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    return (model)

G = define_generator()
D = define_discriminator()
#D.load_weights("model/checkpoint")

def train_discriminator_only(data_size=1000, epochs=200, save_model=False):
    data, label = generate_data(data_size)

    # Fit will actually train the model.
    # X: input of shape (141,2)
    # Y: target catagorization, either 1 or 0. Shape (141,2) for consistancy with X
    D.fit(
        x=np.array(data),
        y=np.array(label),
        epochs=epochs,
        shuffle=True,
        validation_split=0.1,
        use_multiprocessing=True,
        workers=25,
        verbose=2
    )

    # generate testing data
    test_data_size = 1000
    testData, testLabels = generate_data(test_data_size)
    predictions = D.predict(testData)
    fig = plt.figure()
    # We multiply by 4 here because for each dataset, there are 4 points of data.
    plt.hist(abs(predictions.reshape(test_data_size * 4) - testLabels), 50);
    plt.xlim([-0.05, 1.05])

    if save_model:
        D.save_weights('./model/checkpoint')

cross_entropy = BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = Adam(learning_rate = 0.0001)
discriminator_optimizer = Adam(learning_rate = 0.0001)


@tf.function
def train_step(data):
    noise = tf.random.normal([1, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = G(noise, training=True)

        real_output = D(data, training=False)
        fake_output = D(generated_data, training=False)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, G.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, D.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))

    return gen_loss, disc_loss


def train(epochs):
    total_gen_loss = []
    total_disc_loss = []

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        data, _ = generate_data(size=1000)
        gen_loss, disc_loss = train_step(data)
        total_gen_loss.append(gen_loss)
        total_disc_loss.append(disc_loss)

    return np.array(total_gen_loss), np.array(total_disc_loss)

train_discriminator_only(epochs=1000)

total_gen_loss, total_disc_loss = train(1000)

plt.plot(np.arange(0, len(total_gen_loss), 1, int ), total_gen_loss)
plt.plot(np.arange(0, len(total_disc_loss), 1, int ), total_disc_loss)
fig=plt.Figure()
plt.show()

noise = tf.random.normal([1, 100])
generated_data = G(noise, training=False)
generated_data = generated_data.numpy().flatten()
plt.plot(np.arange(0, len(generated_data), 1, int ), generated_data)
fig=plt.Figure()
plt.show()