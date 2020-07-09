from keras import models,optimizers
from keras.layers import Dense, Conv1D,Flatten

def feedForward_classifier(class_count,embedding_size):
    network = models.Sequential()
    network.add(Dense(64, activation='relu', ))
    network.add(Dense(class_count, activation='sigmoid'))
    adm=optimizers.adam()
    network.compile(optimizer=adm,             
                loss='binary_crossentropy')
    return network