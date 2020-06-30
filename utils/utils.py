import pandas as pd
import numpy as np
import os

import pickle
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

    
    
def UniprotID_to_StringId(UniprotIDs):
    
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    uniprot_2_string_file = os.path.join(cur_dir, '../data', 'uniprot_2_string.2018.tsv')
    mapString2Uniprot=pd.read_csv(uniprot_2_string_file,sep='\t',skiprows=1,usecols=[1,2])
    mapString2Uniprot.columns=['uniprot_ac_uniprot_id', 'string_id']
    mapString2Uniprot['uniprot_ac'] = mapString2Uniprot.uniprot_ac_uniprot_id.str.split('|').str[0]
    MergedIDs=pd.merge(UniprotIDs,mapString2Uniprot,on=['uniprot_ac'],how='left')
    print ('Not Matched: ',len(np.nonzero (pd.isna(MergedIDs['string_id']))[0]))
    return MergedIDs

def generate_tripletProt_embeddings(dfTrain):
    
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    proteins_file = os.path.join(cur_dir, '../data/pickles', 'StringIDs.pickle')
    embedding_model_file=os.path.join(cur_dir, '../Saved_Models/', 'trained_model_to_generate_embeddings.h5')
    def identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

    with open(proteins_file, "rb") as f:
        proteins=pickle.load(f)

    lenMax=len(proteins)

    embedding_model = load_model(embedding_model_file,custom_objects={'identity_loss': identity_loss })
    
    embedding_size= 64

    def generate_vector(model, protein_id):

        vector = model.get_layer('embedding').get_weights()[0][protein_id]

        return vector
    
    
    trainProtein_weights = np.zeros((dfTrain.shape[0], embedding_size))
    
    c_found=0
    for i,row in dfTrain.iterrows():
        try:
            protein_id=np.searchsorted(proteins,row['string_id'])
            if protein_id != lenMax:
                c_found += 1
                trainProtein_weights[i]=generate_vector(embedding_model,protein_id)
            else:
                trainProtein_weights[i]=np.random.rand(embedding_size)
        except:
                trainProtein_weights[i]=np.random.rand(embedding_size)


    print('number of found: ',c_found)
    
    del embedding_model
    
    trainProtein_weights=pd.DataFrame(trainProtein_weights)

    return trainProtein_weights


def get_unirep_embeddings(dfTrain,embedding_size):
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    trainProtein_weights = np.zeros((dfTrain.shape[0], embedding_size))
    if embedding_size==1900:
        path = os.path.join(cur_dir, '../data/pickles/embeddings1900_Unirep_withStringIDs.pickle')
        with open(path, "rb") as f:
            embeddings=pickle.load(f)
            identifiers=pickle.load(f)
    
    if embedding_size==256:
        path = os.path.join(cur_dir, '../data/pickles/embeddings256_Unirep_withStringIDs.pickle')
        with open(path, "rb") as f:
            embeddings=pickle.load(f)
            identifiers=pickle.load(f)

    if embedding_size==64:
        path = os.path.join(cur_dir, '../data/pickles/embeddings64_Unirep_withStringIDs.pickle')
        with open(path, "rb") as f:
            embeddings=pickle.load(f)
            identifiers=pickle.load(f)
    not_found=0
    for i in range(dfTrain.shape[0]):  
        try:
            index=identifiers.index(dfTrain.iloc[i][9])
            trainProtein_weights[i]=embeddings[index]
        except:
            trainProtein_weights[i]=np.random.rand(embedding_size)
            not_found+=1
    
    return trainProtein_weights

def generate_unirep_embeddings(train_sequences,embedding_size):
    tf.set_random_seed(42)
    np.random.seed(42)
    embedding_size=64

    cur_dir=os.path.dirname(os.path.abspath(__file__))
    

    if embedding_size==64:

       # Import the mLSTM babbler model
        from unirep import babbler64 as babbler

        # Where model weights are stored.
        
        MODEL_WEIGHT_PATH = os.path.join(cur_dir, '../data/Unirep/64_weights')
        batch_size = 1
        b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)

    elif embedding_size==256:
        # Sync relevant weight files
        #######################  UN COMMENT TO DOWLOAD ##!aws s3 sync --no-sign-request --quiet s3://unirep-public/256_weights/ 256_weights/

        # Import the mLSTM babbler model
        from unirep import babbler256 as babbler

        # Where model weights are stored.
        
        MODEL_WEIGHT_PATH = os.path.join(cur_dir, '../data/Unirep/256_weights')

        batch_size = 1

        b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)

    else:
        # Sync relevant weight files
        #!aws s3 sync --no-sign-request --quiet s3://unirep-public/64_weights/ 64_weights/

        # Import the mLSTM babbler model
        from unirep import babbler1900 as babbler
        # Where model weights are stored.
        MODEL_WEIGHT_PATH = os.path.join(cur_dir, '../data/Unirep/1900_weights/')
        

        batch_size = 1

        b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)

    #actual embedding values will be replaced if is exist
    train_embeddings=np.random.rand(train_sequences.shape[0],embedding_size)

    valid_counts=0
    for i,seq in enumerate(train_sequences):   
        if i%10==0:
            tf.reset_default_graph()        
            del b
            b = babbler(batch_size=1, model_path=MODEL_WEIGHT_PATH)
            print(i)
        if b.is_valid_seq(seq):
            #if lengths[i]<500:
                valid_counts+=1
                #get average hidden states
                train_embeddings[i]=b.get_rep(seq)[0]

    return train_embeddings