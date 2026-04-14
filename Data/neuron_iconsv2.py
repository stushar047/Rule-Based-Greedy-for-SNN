import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris,load_wine, load_breast_cancer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from mravens_icons import *

import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

import json

# Define the file path
json_file = "config_data.json"

# Read the JSON file
with open(json_file, 'r') as file:
    data = json.load(file)

# Extract values from the data dictionary
seed = data["seed"]
f_sample = data["f_sample"]
abs_period_ = data["abs_period_"]
epochs = data["epochs"]
abs_epoch = data["abs_epoch"]
method = data["method"]
app = data["app"]
lr = data["lr"]
b1 = data["b1"]
b2 = data["b2"]
batch_size = data["batch_size"]


# Define the file path
json_file = "config_data.json"

# Read the JSON file
with open(json_file, 'r') as file:
    data = json.load(file)

np.random.seed(seed)

tf.random.set_seed(seed)

def one_hot(y,n):
    
    y_onehot = np.zeros((len(y),n))

    for i,j in enumerate(y):

        y_onehot[i,j]=1
        
    return y_onehot

def train_test_data(application="iris",f_sample=32,seed = 42, test_size=0.20):

    if application=="iris":

        data=load_iris()

        X,y=data['data'],data['target']  
    
    elif application=="wine":
        
        data = load_wine()

        X,y=data['data'],data['target']
        
    elif application=="breast_cancer":
        
        data=load_breast_cancer()

        X,y=data['data'],data['target']

    y_onehot = one_hot(y,np.max(y)+1)
    
    X_norm=(X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))

    X_train,X_test,ytrain,ytest = train_test_split(X_norm,y,test_size=0.20,random_state=seed, stratify=y)
    
    y_train, y_test = one_hot(ytrain,np.max(y)+1), one_hot(ytest,np.max(y)+1)
    
    return X_train,X_test,y_train,y_test, ytrain,ytest

def rate_encoded_data(X,f_sample=32):
    
    X_rate = (X*f_sample).astype('int')

    X_rate_data=np.zeros((X_rate.shape[0],X_rate.shape[1],f_sample))

    for m,sample in enumerate(X_rate):

        for n,feature in enumerate(sample):

            X_rate_data[m,n,:]=np.zeros(f_sample)

            if feature>0:

                X_rate_data[m,n,0:-1:round(f_sample/feature)]=1
                
    return X_rate_data

def get_GD_based_synaptic_weights(X_train,y_train,loss='categorical_crossentropy',activation='softmax',lr=0.1,
                                  b1=0.7,b2=0.999,epochs=200, batch_size=64, verbose=1,seed=42,shuffle=False):

    tf.random.set_seed(seed)

    # Define the input shape
    input_shape = (X_train.shape[1],)  # Replace input_size with the size of your input

    # Create a Sequential model
    model = Sequential()

    initializer = tf.keras.initializers.HeNormal()

    # Add the output layer directly
    model.add(Dense(units=y_train.shape[1], input_shape=input_shape, activation=activation,use_bias=False, kernel_initializer=initializer))

    opt = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=b1,beta_2=b2)

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test,y_test),shuffle=shuffle)

    keras_loss_train, keras_acc_train = model.evaluate(X_train, y_train, verbose=0)

    keras_loss_test, keras_acc_test= model.evaluate(X_test, y_test, verbose=0)

    keras_weights = model.get_weights()[0]
    
    return keras_loss_train, keras_acc_train, keras_loss_test, keras_acc_test, keras_weights

def simulate(X_rate_data,keras_weights,f_sample,st_point=[0,1,2,3],n_id =[0,1,2],
                  type_id=["output","output","output"],abs_period=[1,10,1]):

    y_pred = []
    
    spikes=[]

    for i in range(len(X_rate_data)):

        net = create_custom_network()

        net.add_stimuli(st_point,n_id,X_rate_data[i],keras_weights)

        for (i,j,k) in zip(n_id,type_id,abs_period):

            net.add_neuron(i,j)
            
            net.neuron[i].abs_period=k

        proc=MRAVENS(-5,5)

        event=process_event(f_sample,net,proc)

        event.apply_spike()

        y_pred.append(np.argmax(np.sum(event.spikes_,axis=0)))
        
        spikes.append(np.sum(event.spikes_,axis=0))
        
    return np.array(y_pred), spikes

def train_abs(X_rate_data_train,ytrain,X_rate_data_test,ytest, keras_weights,f_sample,abs_epoch,st_point=[0,1,2,3],
              n_id =[0,1,2],type_id=["output","output","output"],abs_period=np.array([5,5,5]),method="ID"):

    acc_train_list, acc_test_list, abs_period_list = [], [], []

    for i in range(abs_epoch):

        y_pred_train, spiketr = simulate(X_rate_data_train,keras_weights,f_sample,st_point=st_point,n_id =n_id,type_id=type_id,abs_period=abs_period)

        y_pred_test, spikets = simulate(X_rate_data_test,keras_weights,f_sample,st_point=st_point,n_id =n_id,type_id=type_id,abs_period=abs_period)

        acc_train, acc_test = len(ytrain[y_pred_train==ytrain])/len(ytrain), len(ytest[y_pred_test==ytest])/len(ytest)

        print(f"Iteration:{i+1} ::: Neuromorphic Accuracy: Train: {np.round(acc_train,3)},Test: {np.round(acc_test,3)}")

        acc_train_list.append(acc_train)

        acc_test_list.append(acc_test)

        abs_period_list.append(abs_period)

        print(f"Abs_period: {abs_period}")

        spike_dev = np.sum(one_hot(y_pred_train,y_train.shape[1])-y_train,axis=0)

        print(f"Spike Dev: {spike_dev}")

        spike_dev_arr = np.zeros_like(spike_dev)

        idx = np.argsort(spike_dev)
        
        if method =="ID":

            spike_dev_arr[idx[0]], spike_dev_arr[idx[-1]]=  -1,1 
            
        elif method =="I":
            
            spike_dev_arr[idx[0]], spike_dev_arr[idx[-1]]=  0,1 
            
        elif method =="D":
            
            spike_dev_arr[idx[0]], spike_dev_arr[idx[-1]]=  -1,0 

        abs_period = (abs_period + spike_dev_arr).astype('int')

        abs_period[abs_period<0]=0

    acctrain, acctest,absperiod = np.array(acc_train_list), np.array(acc_test_list), np.array(abs_period_list)

    metric = acctrain*0.50 + acctest*45 + 0.05*(acctrain - acctest)  
    
    idx = np.argmax(metric)

    print(f"Best: train: {np.round(acctrain[idx],3)}, test: {np.round(acctest[idx],2)}, abs_period: {absperiod[idx]}")
    
    return acctrain, acctest, absperiod, np.round(acctrain[idx],3), np.round(acctest[idx],2), absperiod[idx]

"""
Train_Test_Data
"""

X_train,X_test,y_train,y_test,ytrain,ytest=train_test_data(application=app,f_sample=f_sample,seed=seed, test_size=0.20)

"""
Keras Weights
"""

# Define the file path
json_file = f"{app}_model.json"

# Read the JSON file
with open(json_file, 'r') as file:
    model_data = json.load(file)


keras_loss_train, keras_acc_train, keras_loss_test, keras_acc_test, keras_weights = model_data["keras_loss_train"], model_data["keras_acc_train"], model_data["keras_loss_test"], model_data["keras_acc_test"], np.array(model_data["keras_weights"])

"""
Rate encoded data
"""

X_rate_data_train, X_rate_data_test = rate_encoded_data(X_train,f_sample=f_sample), rate_encoded_data(X_test,f_sample=f_sample)

print(f"Keras Accuracy: Train: {np.round(keras_acc_train,3)}, Test: {np.round(keras_acc_test,3)}")

abs_period = np.ones(y_train.shape[1])*abs_period_

_,_,_,acctrain, acctest, absperiod = train_abs(X_rate_data_train,ytrain,X_rate_data_test,ytest, keras_weights,f_sample,abs_epoch,st_point=np.arange(X_train.shape[1]),
              n_id =np.arange(y_train.shape[1]),type_id=["output"]*y_train.shape[1],abs_period=abs_period, method = method)

file=open("result_data1.csv","a")

file.write(f"{f_sample},{abs_period_},{epochs},{abs_epoch},{method},{app},{keras_acc_train},{keras_acc_test},{acctrain},{acctest},{absperiod}\n")

file.close()