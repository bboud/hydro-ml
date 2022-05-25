import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_crossentropy
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%%%
def Kernel(x, x0):
    sigma = 0.8
    protonFraction = 0.4
    norm = protonFraction/(np.sqrt(2.*np.pi)*sigma)
    return(norm*np.exp(-(x - x0)**2./(2.*sigma**2.)))

def testDataGen():
    A = 197
    yBeam = 5.36
    slope = 0.5
    sigmaEtas = 0.2
    
    # generate input data
    nBaryons = np.random.randint(0, 2*A)
    randX = np.random.uniform(0, 1, size=nBaryons)
    etasBaryon = 1./slope*np.arcsinh((2.*randX - 1)*np.sinh(slope*yBeam))
    etasArr = np.linspace(-7, 7, 141)
    dNBdetas = np.zeros(len(etasArr))
    norm = 1./(np.sqrt(2.*np.pi)*sigmaEtas)
    for iB in etasBaryon:
        dNBdetas += norm*np.exp(-(etasArr - iB)**2./(2.*sigmaEtas**2.))
    
    # generate test data with convolution with a kernel
    dNpdy = np.zeros(len(etasArr))
    detas = etasArr[1] - etasArr[0]
    for i in range(len(etasArr)):
        dNpdy[i] = sum(Kernel(etasArr, etasArr[i])*dNBdetas)*detas
        
    # generate fake data with random noise
    dNBdetasFake = np.random.uniform(0.0, dNBdetas.max(), size=len(etasArr))
    dNpdyFake = np.random.uniform(0.0, dNpdy.max(), size=len(etasArr))
    return(etasArr, dNBdetas, dNpdy, dNBdetasFake, dNpdyFake)
#%%%

def generateData(size=500):
    print('Generating Data...')
    dataArr = []
    labelArr = []  
    for iev in range(size):
        x, y1, y2, y3, y4 = testDataGen()
        
        dim = len(x)
        
        # real data
        x = np.hstack((y1.reshape(dim, 1), y2.reshape(dim, 1)))
        dataArr.append(x)
        labelArr.append(np.ones((dim, 2)))
        
        # fake data
        x = np.hstack((y3.reshape(dim, 1), y4.reshape(dim, 1)))
        dataArr.append(x)
        labelArr.append(np.zeros((dim, 2)))
    print("done")
    return(np.array(dataArr), np.array(labelArr))

def defineModel(shape):
    #Sequential model
    model = Sequential()
    
    print(shape)
    
    # #Input layer
    model.add(Dense(25, activation='relu', input_shape=shape))

    #Hidden Layer
    model.add(Dense(10, activation='relu', input_shape=shape))

    #Output layer - Binary
    model.add(Dense(2, activation='sigmoid', input_shape=shape))

    #Adam seems to be the best model. Lowest loss value of about 0.2.
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    model.summary()
    
    return model

def train(epochs=5):
    #Generate the shuffled data
    data, labels = generateData(size=5000)
    
    shape = data[0].shape
    
    model = Sequential([
        Dense(units=32, activation='relu', input_shape=shape),
        Dense(units=16, activation='relu'),
        Dense(units=4, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.summary()
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
                  
    model.fit(
      x=np.array(data),
      y=np.array(labels),
      epochs=epochs,
      use_multiprocessing=True,
      workers=10
    )
    
    model.save_weights('./weights', overwrite=True)
    
train(epochs=200)

def graph(etas, data1, data2):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(10,5))

    #x = np.arange(0,len(data_formatted[0]))
    ax0.scatter(etas, data1*100)
    ax0.scatter(etas, data2*100)
    ax0.set_title('Accuracy over the Distrobutions')
    ax0.set_xlabel(r'$\eta$')
    ax0.set_ylabel('Accuracy %')

    data_formatted = data[5].reshape(2,141)
    
    ax1.plot(etas, data1)
    ax1.set_xlabel(r'$\eta$')
    ax1.set_ylabel('Density')
    
    ax2.plot(etas, data2, color='orange')
    ax2.set_xlabel(r'$\eta$')
    ax2.set_ylabel('PRD')
    
    fig.tight_layout()
    plt.show()

#Data coming out doesn't look correct.
def prediction():
    model = defineModel()
    
    model.load_weights('./weights')
    
    #Data coming out of this function is formatted for the input layer, we need to get it back into its import form.
    etas, data, _ = generateData(size=2)