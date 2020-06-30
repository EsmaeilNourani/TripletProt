from keras import models,optimizers
from keras.layers import Dense, Conv1D,Flatten

def naive_CNN_classifier(class_count,embedding_size):
    network = models.Sequential()
    network.add(Conv1D(8, 7, activation='relu', input_shape=(embedding_size, 1),padding='valid'))
    network.add(Dense(16, activation='relu', ))
    network.add(Flatten())    
    network.add(Dense(class_count, activation='sigmoid'))
    adm=optimizers.adam(lr=0.0001)
    network.compile(optimizer=adm,             
                loss='binary_crossentropy')
    return network