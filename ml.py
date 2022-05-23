from keras.models import Sequential
from keras.layers import Dense
import os
import numpy as np
import tensorflow as tf

data = []
data_y = []

#Size defines how many events in the dataset we want to load.
def generateData(size=500):
    print('Generating Data...')
    
    path = './3DAuAu200_minimumbias_BG16_tune17'
    
    #Cannot enumerate within the for loop becasue it will count all files found and not the number of files we are importing.
    i = 0
    
    for file in os.listdir(path):
        #We want to break this loop when we hit the size we want.
        if i >= size:
            break
        
        #Only register events with baryon_etas
        if file.find('baryon_etas') == -1:
            continue
        
        #This split will get the eventID specifically from the file string.
        eventID = file.split('_')[1]
        
        #Baryons
        _, X1 = np.loadtxt(path+'/event_' + eventID + '_net_baryon_etas.txt', unpack=True)
        
        #Protons
        _, X2, _ = np.loadtxt(path+'/event_' + eventID + '_net_proton_eta.txt', unpack=True)
    
        #Stack neatly
        #The ones and zeros arrays need to be the same shape as the input. We pair them here
        #so that the training data can be shuffled.
        x = np.hstack( ( X1.reshape(141,1), X2.reshape(141,1)))
    
        data.append(x)
        data_y.append(np.ones((141,2)))
        
        #Generate Fake Data
        
        #Baryons
        X1 = np.random.uniform(0.0, 100.0, size=141)
        #Protons
        X2 = np.random.uniform(0.0, 20, size=141)
        # stack arrays
        x = np.hstack((X1.reshape(141, 1), X2.reshape(141,1)))
            
        data.append(x)
        data_y.append(np.zeros((141,2)))
        
        i+=1
        
    print('Done!')

def train(epochs=5):
    #Generate the shuffled data
    generateData(size=2000)
    
    #Sequential model
    model = Sequential()
    
    #Input layer
    model.add(Dense(25, activation='relu', input_shape=(141,2)))
    
    #Hidden Layer
    model.add(Dense(5, activation='relu', input_shape=(141,2)))
    
    #Output layer - Binary
    model.add(Dense(2, activation='sigmoid', input_shape=(141,2)))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    #Fit will actually train the model.
    # X: input of shape (141,2)
    # Y: target catagorization, either 1 or 0. Shape (141,2) for consistancy with X
    model.fit(
        x=np.array(data),
        y=np.array(data_y),
        epochs=epochs,
        use_multiprocessing=True,
        workers=25
    ) 
train(epochs=500)