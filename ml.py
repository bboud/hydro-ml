from keras.models import Sequential
from keras.layers import Dense
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Size defines how many events in the dataset we want to load.
def generateData(size=500):
    print('Generating Data...')
    
    path = './3DAuAu200_minimumbias_BG16_tune17'
    
    #Cannot enumerate within the for loop becasue it will count all files found and not the number of files we are importing.
    i = 0
    
    data = []
    data_y = []
    
    etas = []
    
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
        etas, X1 = np.loadtxt(path+'/event_' + eventID + '_net_baryon_etas.txt', unpack=True)
        
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
    return etas, data, data_y
    
#generateData(size=5)

def defineModel():
    #Sequential model
    model = Sequential()
    
    shape = (141,2)
    
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
    _, data, data_y = generateData(size=2048)
    
    model = defineModel()
    
    # print(np.shape(data))
    # print(np.shape(data_y))
    
    #Fit will actually train the model.
    # X: input of shape (2,141)
    # Y: target catagorization, either 1 or 0. Shape (141,2) for consistancy with X
    model.fit(
        x=np.array(data),
        y=np.array(data_y),
        epochs=epochs,
        use_multiprocessing=True,
        workers=10
    )
    
    model.save_weights('./weights', overwrite=True)
    
#train(epochs=10)

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
    

    
prediction()
# prediction()
# prediction()
# prediction()