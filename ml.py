from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from matplotlib import colors
from matplotlib import pyplot
import os
import numpy as np
import matplotlib.pyplot as plt

#All event numbers
events = []

def initializeSamples():
    #Find all files in the test data directory
    for file in os.listdir('./train_data'):
        #Only register events with baryon_etas
        if file.find('baryon_etas') == -1:
            continue
        #Grab all event numbers
        #print(file.split('_'))
        events.append(file.split('_')[1])
    
initializeSamples()

# generate randoms sample from x^2
def generateFakeSamples():
    #Baryons
    X1 = np.random.uniform(0.0, 100.0, size=141)
    #Protons
    X2 = np.random.uniform(0.0, 20, size=141)
    # stack arrays
    x = np.hstack((X1.reshape(141, 1), X2.reshape(141,1)))
        
    y = np.zeros((141, 1))
            
    return np.array(x), y

def generateRealSamples(e):
    _, X1 = np.loadtxt('./train_data/event_' + events[e] + '_net_baryon_etas.txt', unpack=True)
    _, X2, _ = np.loadtxt('./train_data/event_' + events[e] + '_net_proton_eta.txt', unpack=True)
    
    x = np.hstack( ( X1.reshape(141,1), X2.reshape(141,1) ))
    
    y = np.ones((141, 1))
    return x, y

# define the standalone discriminator model
def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform', input_dim=2))
    #The discriminator will identify real or fake results from what we provide.
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define the discriminator model
model = define_discriminator()
# summarize the model
model.summary()

def train_discriminator(model, n_epochs=1000):
    
    eventIndex = 0
    
    # run epochs manually
    for i in range(n_epochs):
        # generate real examples
        X_real, y_real = generateRealSamples(i)
        # update model
        model.train_on_batch(X_real, y_real)
        # generate fake examples
        X_fake, y_fake = generateFakeSamples()
        # update model
        model.train_on_batch(X_fake, y_fake)
        # evaluate the model
        _, acc_real = model.evaluate(X_real, y_real, verbose=0)
        _, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)
        print(i, acc_real, acc_fake)
        eventIndex += 10
        
train_discriminator(model)
    
#
#
# def define_generator(latent_dim, n_outputs=2):
#     model = Sequential()
#     model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
#     model.add(Dense(n_outputs, activation='linear'))
#     return model
#
# # generate randoms sample from x^2
# def generateLatentSamples():
#     #Baryons
#     X1 = np.random.uniform(0.0, 100.0, size=141)
#     # stack arrays
#     x = X1.reshape(141, 1)
#
#     return x