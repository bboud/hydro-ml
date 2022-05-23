from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import os
import numpy as np

samples_x = []
samples_y = []

def generateData():
    print('Generating Data...')
    
    path = './3DAuAu200_minimumbias_BG16_tune17'
    
    for file in os.listdir(path):
        #Only register events with baryon_etas
        if file.find('baryon_etas') == -1:
            continue
        
        #This split will get the eventID specifically from the file string.
        eventID = file.split('_')[1]
        
        _, X1 = np.loadtxt(path+'/event_' + eventID + '_net_baryon_etas.txt', unpack=True)
        _, X2, _ = np.loadtxt(path+'/event_' + eventID + '_net_proton_eta.txt', unpack=True)
    
        x = np.hstack( ( X1.reshape(141,1), X2.reshape(141,1) ))
    
        y = np.ones((141, 2))
        samples_x.append(np.array(x))
        samples_y.append(y)
        
        #Generate Fake Data as well
        
        #Baryons
        X1 = np.random.uniform(0.0, 100.0, size=141)
        #Protons
        X2 = np.random.uniform(0.0, 20, size=141)
        # stack arrays
        x = np.hstack((X1.reshape(141, 1), X2.reshape(141,1)))
        
        y = np.zeros((141, 2))
            
        samples_x.append(np.array(x))
        samples_y.append(y)
        
        
    print('Done!')
    
generateData()

# define the standalone discriminator model
def discriminate(epochs=5):
    model = Sequential()
    model.add(Dense(25, activation='relu', input_shape=(141,2)))
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(5, activation='relu'))
    #The discriminator will identify real or fake results from what we provide.
    model.add(Dense(2, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=np.array(samples_x),y=np.array(samples_y), epochs=epochs, use_multiprocessing=True, workers=25)

    print(model.predict(x=np.array(samples_x), verbose=1)[27], samples_x[27])
    print(model.predict(x=np.array(samples_x), verbose=1)[300], samples_x[300])
    print(model.predict(x=np.array(samples_x), verbose=1)[146], samples_x[146])
    
    model.summary()
    
discriminate(epochs=100)