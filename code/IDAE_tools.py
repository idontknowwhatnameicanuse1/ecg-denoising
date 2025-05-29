import numpy as np

import soft_threshold_WT_tools as WT_tls
# from model import DNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

def file_path(servier):
    if servier=='autodl':
        path_name='/root/autodl-tmp/ddpm'
    if servier=='hust_individual':
        path_name='/root/IDAE'
    if servier=='hust_cluster':
        path_name='/home/uu201912628/DDPM'
    return path_name

def DNN():
    model=Sequential()
    # model.input.shape=(none,1024,1)
    model.add(Dense(units=1034,activation='sigmoid'))
    model.add(Dense(units=517,activation='sigmoid'))
    model.add(Dense(units=517,activation='sigmoid'))
    model.add(Dense(units=1034,activation='sigmoid'))
    # shape=(none,1024,1)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def IDAE_denoise_train(path_name,train_clean,train_noisy,
                       batch_size,epoch):
    WT_train_noisy=WT_tls.dwt_soft_threshold(input=train_noisy)
    # reshape
    print('reshape')
    WT_train_noisy = np.reshape(WT_train_noisy, (WT_train_noisy.shape[0]*WT_train_noisy.shape[2], WT_train_noisy.shape[1]))
    train_clean = np.reshape(train_clean, (train_clean.shape[0]*train_clean.shape[2], train_clean.shape[1]))
    print('training')
    model=DNN()
    # model train
    model.fit(x=WT_train_noisy,y=train_clean,batch_size=batch_size,epochs=epoch)
    model.save(path_name + '/saved_model/idae_model')

def IDAE_denoise(path_name, testset_noisy):
    model=tf.keras.models.load_model(path_name+'/saved_model/idae_model')
    model.compile()
    # model valid
    denoise_data = np.zeros(testset_noisy.shape)
    WT_test_noisy=WT_tls.dwt_soft_threshold(input=testset_noisy)
    for j in range(WT_test_noisy.shape[-1]):
        denoise_data[:, :, j]=model.predict(WT_test_noisy[:, :, j])
    return denoise_data
