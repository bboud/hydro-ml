from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.losses import BinaryCrossentropy

#JIT for the data generation.
from numba import jit

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

@jit
def Kernel(x, x0):
    sigma = 0.8
    protonFraction = 0.4
    norm = protonFraction/(np.sqrt(2.*np.pi)*sigma)
    return(norm*np.exp(-(x - x0)**2./(2.*sigma**2.)))

@jit
def FakeKernel(x, x0, s):
    sigma = s #Just a slight nudge
    protonFraction = 0.4
    norm = protonFraction/(np.sqrt(2.*np.pi)*sigma)
    return(norm*np.exp(-(x - x0)**2./(2.*sigma**2.)))

@jit
def test_data_gen(fakeKernel=False, sigma=0.4):
    A = 197
    yBeam = 5.36
    slope = 0.5
    sigmaEtas = 0.2
    
    # generate input data
    nBaryons = np.random.randint(0, 2*A)
    randX = np.random.uniform(0, 1, size=nBaryons)
    etasBaryon = 1./slope*np.arcsinh((2.*randX - 1)*np.sinh(slope*yBeam))
    etasArr = np.linspace(-6.4, 6.4, 128)
    dNBdetas = np.zeros(len(etasArr))
    norm = 1./(np.sqrt(2.*np.pi)*sigmaEtas)
    for iB in etasBaryon:
        dNBdetas += norm*np.exp(-(etasArr - iB)**2./(2.*sigmaEtas**2.))
    
    # generate test data with convolution with a kernel
    dNpdy = np.zeros(len(etasArr))
    detas = etasArr[1] - etasArr[0]
    for i in range(len(etasArr)):
        dNpdy[i] = sum(Kernel(etasArr, etasArr[i])*dNBdetas)*detas
    
    if fakeKernel:
        #generate test data with convolution with a fake kernel
        dNBdetasFake = np.random.uniform(0.0, dNBdetas.max(), size=len(etasArr))
        dNpdyFake = np.zeros(len(etasArr))
        detas = etasArr[1] - etasArr[0]
        for i in range(len(etasArr)):
            dNpdyFake[i] = sum(FakeKernel(etasArr, etasArr[i], sigma)*dNBdetas)*detas
            
        return(etasArr, dNBdetas, dNpdy, dNBdetas, dNpdyFake)
    else:
        # generate fake data with random noise
        dNBdetasFake = np.random.uniform(0.0, dNBdetas.max(), size=len(etasArr))
        dNpdyFake = np.random.uniform(0.0, dNpdy.max(), size=len(etasArr))

        return(etasArr, dNBdetas, dNpdy, dNBdetasFake, dNpdyFake)
    
#Depth defines the number of blocks of data that will be returned per "iev".
# BATCH_SIZE % DEPTH must equal 0 otherwise you will get a weird datashape due to the casing of the division.
DEPTH = 2

def generate_data(size=128):
    #Tensor that is returned
    dataArr = []
    labelArr = []
    
    #We need to build a discrete tensor to append.
    for iev in range( int(size/DEPTH) ):
        x, y1, y2, y3, y4 = test_data_gen(fakeKernel=False)
        
        # real data - Block
        x = y2
        dataArr.append(x)
        labelArr.append(1)
        
        # fake data: random - Block
        x = y4
        dataArr.append(x)
        labelArr.append(0)
        
    #data = np.array(dataArr, dtype=np.float32)
        
    return(np.array(dataArr, dtype=np.float32), np.array(labelArr))

#Global Constants

#Discriminator
DFILTER = 32
DKERNEL = "he_normal"
DROPOUT_RATE = 0.4

#Generator
BATCH_SIZE = 128
LATENT_DIM = 16
KERNEL_SIZE = 7
KERNEL = "ones"

#Feature maps
FILTER = 128

def define_discriminator(dimShape=(128,1)):
    #Sequential model
    model = Sequential([
        layers.Input(shape=dimShape),
        
        layers.Conv1D(DFILTER, 4, strides = 2, activation="LeakyReLU", padding="same", kernel_initializer=DKERNEL),
        layers.Dropout(DROPOUT_RATE),
        layers.Conv1D(DFILTER, 4, strides = 2, activation="LeakyReLU", padding="same", kernel_initializer=DKERNEL),
        layers.Dropout(DROPOUT_RATE),
        layers.Conv1D(DFILTER, 4, strides = 2, activation="LeakyReLU", padding="same", kernel_initializer=DKERNEL),
        layers.Dropout(DROPOUT_RATE),
        layers.Conv1D(DFILTER, 4, strides = 2, activation="LeakyReLU", padding="same", kernel_initializer=DKERNEL),
        
        layers.GlobalMaxPool1D(),
        
        layers.Dense(1, activation="sigmoid")
    ])
    model.summary()

    return(model)

def define_generator():

    model = Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        #We then reshape this into 2D space of size
        layers.Dense(LATENT_DIM * FILTER, input_dim=FILTER, activation="LeakyReLU"),
        layers.Reshape((LATENT_DIM, FILTER)),
        
        #From here, we work to size up to the output shape that we want.
        layers.Conv1DTranspose(FILTER, KERNEL_SIZE, strides = 2, padding="same", kernel_initializer=KERNEL),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv1DTranspose(FILTER, KERNEL_SIZE, strides = 2, padding="same", kernel_initializer=KERNEL),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv1DTranspose(FILTER, KERNEL_SIZE, strides = 2, padding="same", kernel_initializer=KERNEL),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv1D(1,1, activation="sigmoid"),
    ])

    model.summary()

    return(model)

D = define_discriminator()
G = define_generator()

cross_entropy = BinaryCrossentropy(from_logits=True)

generator_optimizer = Adam(learning_rate = 0.0001)
discriminator_optimizer = Adam(learning_rate = 0.00001)

#Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
@tf.function
def train_step(data):
    fake = G(tf.random.normal(shape=(BATCH_SIZE, LATENT_DIM)))
    
    combined = tf.concat([tf.reshape(fake, (BATCH_SIZE, data.shape[0],)), data], axis=0)
    
    labels = tf.concat([tf.ones((BATCH_SIZE, 1)), tf.zeros((data.shape[0], 1))], axis=0)
    
    labels += 0.05 * tf.random.uniform(labels.shape)
    
    #https://www.tensorflow.org/guide/autodiff
    with tf.GradientTape() as disc_tape:
        predictions = D(combined, training=True)
        d_loss = cross_entropy(labels, predictions)
    
    gradients_of_discriminator = disc_tape.gradient(d_loss, D.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))
    
    wrong_labels = tf.zeros((BATCH_SIZE, 1))
    
    with tf.GradientTape() as gen_tape:
        predictions = D(G(tf.random.normal((BATCH_SIZE, LATENT_DIM,)), training = True))
        g_loss = cross_entropy(wrong_labels, predictions)
    
    gradients_of_generator = gen_tape.gradient(g_loss, G.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
    
    return g_loss, d_loss

def train(epochs):
    total_gen_loss = []
    total_disc_loss = []
    
    for epoch in range(epochs):
        print("Epoch: " + str(epoch+1))
        data, _ = generate_data(size=BATCH_SIZE)
        gen_loss, disc_loss = train_step(data)
        total_gen_loss.append(gen_loss)
        total_disc_loss.append(disc_loss)
    return np.array(total_gen_loss), np.array(total_disc_loss)

total_gen_loss, total_disc_loss = train(500)

fig, axs = plt.subplots(1, 2, figsize=(4, 2), constrained_layout=True)
axs[0].plot(np.arange(0, len(total_gen_loss), 1, int ), total_gen_loss)
axs[0].plot(np.arange(0, len(total_disc_loss), 1, int ), total_disc_loss)

noise = tf.random.normal([1, LATENT_DIM])
generated_data = G(noise, training=False)
print(D.predict(generated_data))
generated_data = generated_data.numpy().flatten()
axs[1].plot(np.arange(0, len(generated_data), 1, int ), generated_data)
plt.show()