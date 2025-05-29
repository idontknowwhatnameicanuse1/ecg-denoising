import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tqdm import tqdm
import os
import tensorflow.keras.metrics
# from tensorflow.keras import mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

def match_Datagen(Data_clean_ECG, Data_noise_BW, Data_noise_EM, Data_noise_MA, SNR, mix_por):
    # add the noise to clean ECG to create match data
    # noise gen
    # length=Data_clean_ECG.shape[0]
    # Data_clean_ECG=np.sin(np.arange(length)*2*np.pi/360)
    # Data_clean_ECG=np.reshape(Data_clean_ECG,(length,1))
    noise=(Data_noise_BW*mix_por[0]+Data_noise_EM*mix_por[1]+Data_noise_MA*mix_por[2])/np.sum(mix_por)
    noise=noise[:Data_clean_ECG.shape[0],:]
    noise=noise# 1.2
    # noise=np.random.normal(size=(Data_clean_ECG.shape[0],Data_clean_ECG.shape[1]))
    # SNR adjust
    k=(np.sqrt(np.sum(Data_clean_ECG**2)/(np.sum(noise**2)*np.power(10,(SNR/10))))).astype('double')
    # match data gen
    Data_noisy = Data_clean_ECG+k*noise
    Data_noisy=(Data_noisy-np.min(Data_noisy))/\
               (np.max(Data_noisy)-np.min(Data_noisy))
    Data_clean_ECG = (Data_clean_ECG - np.min(Data_clean_ECG)) / \
                 (np.max(Data_clean_ECG) - np.min(Data_clean_ECG))
    # baseline=np.real(np.fft.fftshift(np.fft.fft(Data_clean_ECG))[0,0])
    # Data_clean_ECG=Data_clean_ECG-baseline

    # Data_noisy=(Data_noisy-np.mean(Data_noisy))/np.std(Data_noisy)
    # Data_clean_ECG=(Data_clean_ECG-np.mean(Data_clean_ECG))/(3*np.std(Data_clean_ECG))

    match_sample_long=np.concatenate((Data_clean_ECG,Data_noisy),axis=1)
    return match_sample_long

def longdata_div(long_data,div_size):
    # divide the long match data into pieces
    T=long_data.shape[0]
    piece_num=(np.floor(T/div_size)).astype('int')
    match_sample=np.zeros((piece_num,div_size,2))
    recover_bar=np.zeros((piece_num,1))
    for i in range(piece_num):
        x0=long_data[i*div_size:(i+1)*div_size,0]
        x1 = long_data[i * div_size:(i + 1) * div_size, 1]
        x0=np.reshape(x0,(div_size,1))
        x1=np.reshape(x1,(div_size,1))

        recover_bar[i,0]=np.real((1/div_size*np.fft.fft(x0,axis=0))[0])
        x0=x0-np.real((1/div_size*np.fft.fft(x0,axis=0))[0])
        match_sample[i, :, 0]=np.reshape(x0,(div_size))

        x1=x1-np.real((1/div_size*np.fft.fft(x1,axis=0))[0])
        match_sample[i, :, 1]=np.reshape(x1,(div_size))
        # plt.plot(x0)
        # plt.show()
        # plt.plot(x1)
        # plt.show()
    return match_sample,recover_bar

def longdata_div_multi_channel(long_data,div_size):
    # divide the long match data into pieces
    T=long_data.shape[0]
    extra_windows=3
    piece_num=(np.floor(T/div_size)).astype('int')
    match_sample=np.zeros((piece_num-extra_windows,div_size,2+extra_windows))
    for i in range(extra_windows,piece_num):
        k=i-extra_windows
        match_sample[k,:,:2]=long_data[i*div_size:(i+1)*div_size,:]
        extra_noisy_signal=np.reshape(long_data[(i-extra_windows)*div_size:i*div_size,1]
                                      ,(div_size,extra_windows))
        match_sample[k,:,2:2+extra_windows]=extra_noisy_signal
    return match_sample

def mit_data_read_gen(div_size,SNR,mix_por=np.array([1,0,0])):
    # data path
    # DataDir_clean_ECG ="/home/zhanghaowei/project/denoiseCGAN/mit_ECG_csv/"
    # DataDir_noise_BW = "/home/zhanghaowei/project/denoiseCGAN/mit_noise_csv/"
    # DataDir_noise_EM = "/home/zhanghaowei/project/denoiseCGAN/mit_noise_csv/"
    # DataDir_noise_MA = "/home/zhanghaowei/project/denoiseCGAN/mit_noise_csv/"

    DataDir_clean_ECG ="/home/uu201912628/project/mit_ECG_csv/"
    DataDir_noise_BW = "/home/uu201912628/project/mit_noise_csv/"
    DataDir_noise_EM = "/home/uu201912628/project/mit_noise_csv/"
    DataDir_noise_MA = "/home/uu201912628/project/mit_noise_csv/"

    # DataDir_clean_ECG ="/root/denois_CGAN/mit_ECG_csv/"
    # DataDir_noise_BW = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_EM = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_MA = "/root/denois_CGAN/mit_noise_csv/"

    # noise reading
    Data_noise_BW = pd.read_csv((DataDir_noise_BW+'bw.csv'))
    Data_noise_EM = pd.read_csv((DataDir_noise_EM+'em.csv'))
    Data_noise_MA = pd.read_csv((DataDir_noise_MA+'ma.csv'))
    Data_noise_BW = Data_noise_BW.to_numpy()
    Data_noise_EM = Data_noise_EM.to_numpy()
    Data_noise_MA = Data_noise_MA.to_numpy()

    # clean ECG reading & match data generation
    match_sample=[]
    filenames=os.listdir(DataDir_clean_ECG)
    nfile=len(filenames)
    for i in range(nfile):
        fulname=(DataDir_clean_ECG+filenames[i])
        Data_clean_ECG=pd.read_csv(fulname)
        Data_clean_ECG=Data_clean_ECG.to_numpy()

        # randomly choose a col in noise data as noise
        col_rand_array = np.arange(Data_noise_BW.shape[1])
        np.random.shuffle(col_rand_array)
        BW = Data_noise_BW[:,col_rand_array[0]]
        # BW = Data_noise_BW[:, 0]
        BW=np.reshape(BW,(Data_clean_ECG.shape[0],1))

        col_rand_array = np.arange(Data_noise_EM.shape[1])
        np.random.shuffle(col_rand_array)
        EM = Data_noise_EM[:,col_rand_array[0]]
        # EM = Data_noise_EM[:, 0]
        EM=np.reshape(EM,(Data_clean_ECG.shape[0],1))

        col_rand_array = np.arange(Data_noise_MA.shape[1])
        np.random.shuffle(col_rand_array)
        MA = Data_noise_MA[:,col_rand_array[0]]
        # MA = Data_noise_MA[:, 0]
        MA=np.reshape(MA,(Data_clean_ECG.shape[0],1))

        # match data generation
        match_sample_long=match_Datagen(Data_clean_ECG, BW, EM, MA,
                      SNR, mix_por)
        sample=longdata_div(match_sample_long,div_size)  # match_sample.shape=(sample_num,time,type)
        if match_sample==[]:
            match_sample=sample
        else:
            match_sample=np.concatenate((match_sample,sample),axis=0)
    return match_sample

def data_batchlization(data,batchsize):
    # divide the dataset into batches, left out the final samples which not a full batch
    sample_num=data.shape[0]
    datadim=data.shape[1]
    channel=data.shape[2]
    batch_num=np.floor(sample_num/batchsize).astype('int')
    batched_data=np.zeros((batch_num,batchsize,datadim,channel))
    for i in range(batch_num):
        batched_data[i,:,:,:]=data[i*batchsize:(i+1)*batchsize,:,:]
    return batched_data

def data_batchlization_gen_noise_try(data,batchsize,SNR):
    # divide the dataset into batches, left out the final samples which not a full batch
    sample_num=data.shape[0]
    datadim=data.shape[1]
    channel=data.shape[2]
    batch_num=np.floor(sample_num/batchsize).astype('int')
    # gen noise
    gen_core=tf.keras.models.load_model('/root/autodl-tmp/project/gen_noise_core/G')

    batched_data=np.zeros((batch_num,batchsize,datadim,channel))
    for i in range(batch_num):
        batched_data[i,:,:,0]=data[i*batchsize:(i+1)*batchsize,:,0]
        Data_clean_ECG=batched_data[i,:,:,0]
        Data_clean_ECG=np.reshape(Data_clean_ECG,(batchsize,datadim,1))
        gauss=np.random.normal(0,1,size=(batchsize,datadim,1))
        gen_noise_input=np.concatenate([Data_clean_ECG,gauss],axis=2)
        noise=gen_core.predict(gen_noise_input)

        k = (np.sqrt(np.sum(Data_clean_ECG ** 2) / (np.sum(noise ** 2) * np.power(10, (SNR / 10))))).astype('double')
        Data_noisy=Data_clean_ECG+k*noise
        Data_noisy=(Data_noisy-np.min(Data_noisy))\
                   /(np.max(Data_noisy)-np.min(Data_noisy))
        batched_data[i, :, :, 1]=np.squeeze(Data_noisy)
    return batched_data

def recording_norm(long_match_data):
    # normalization for every sample, using max and min norm
    length=long_match_data.shape[0]
    channel=long_match_data.shape[1]
    normed_data=np.zeros((length,channel))
    # normed_recov=np.zeros((sample_num,2))
    normed_reconv=0
    clean_data=long_match_data[:,0]
    noisy_data = long_match_data[:, 1]
    normed_data[:, 0] = 2 * (clean_data - np.min(clean_data)) / (
                np.max(clean_data) - np.min(clean_data)) - 1
    normed_data[:, 1] = 2 * (noisy_data - np.min(noisy_data)) / (
                np.max(noisy_data) - np.min(noisy_data)) - 1
    # normed_recov[i, 0] = np.min(noisy_sample)
    # normed_recov[i, 1] = np.max(noisy_sample)
    return normed_data,normed_reconv

def piece_norm(dataset):
    # normalization for every sample, using max and min norm
    sample_num=dataset.shape[0]
    datadim=dataset.shape[1]
    channel=dataset.shape[2]
    normed_sample=np.zeros((sample_num,datadim,channel))
    normed_recov=np.zeros((sample_num,2))
    for i in range(sample_num):
        clean_sample=dataset[i,:,0]
        normed_sample[i,:,0]=(clean_sample-np.min(clean_sample))\
                             /(np.max(clean_sample)-np.min(clean_sample))
        # normed_sample[i,:,0]=clean_sample
        for j in range(channel):
            noisy_sample=dataset[i,:,j]
            normed_sample[i,:,j]=(noisy_sample-np.min(noisy_sample))\
                                 /(np.max(noisy_sample)-np.min(noisy_sample))

        normed_recov[i,0]=np.min(clean_sample)
        normed_recov[i,1]=np.max(clean_sample)
    return normed_sample,normed_recov

def piece_denorm(normed_sample,normed_recov):
    sample_num=normed_sample.shape[0]
    datadim=normed_sample.shape[1]
    channel=normed_sample.shape[2]
    denormed_sample=np.zeros((sample_num,datadim,channel))
    for i in range(sample_num):
        min=normed_recov[i,0]
        max=normed_recov[i,1]
        denormed_sample[i,:,:]=min+denormed_sample[i,:,:]*(max-min)
    return denormed_sample

def batch_conv(input,kernel,padding='same'):
    num=input.shape[0]   # input.shape=(num,div_size,channel=1)
    div_size=input.shape[1]
    result=np.zeros((num,div_size))
    conv_kernel=np.squeeze(kernel)
    conv_input=np.squeeze(input)
    for i in range(num):
        result[i,:]=np.convolve(conv_input[i,:],conv_kernel,mode='same')
        # result[i,:]=np.convolve(result[i,:],conv_kernel,mode='same')
        # result[i, :] = np.convolve(result[i, :], conv_kernel, mode='same')
        # result[i, :] = np.convolve(result[i, :], conv_kernel, mode='same')
        # result[i,:] = np.exp(-100 * (result[i,:] - np.mean(result[i,:]))**2)
    result=np.reshape(result,(num,div_size,1))
    # result=result-threshold
    # result[result<0]=0

    return result

def kernel_gen(kernel_size,sigma,amp):
    kernel=np.zeros((kernel_size,1))
    miu=int(kernel_size/2)
    for i in range(kernel_size):
        kernel[i,:]=amp*np.exp(-(i-miu)**2/(2*sigma**2))
    # for i in range(kernel_size):
    #     kernel[i,:]=amp-abs(2*amp/kernel_size*i-amp)
    return kernel

def dataset_gen(match_sample,recover_bar,train_por):
    # generate trainset and testset from total dataset
    num=match_sample.shape[0]
    train_num=np.floor(num*train_por).astype('int')

    # randomly choose sample from total data
    index_rand_array = np.arange(match_sample.shape[0])
    np.random.shuffle(index_rand_array)
    # trainset = match_sample[index_rand_array[:train_num],:,:]
    # testset=match_sample[index_rand_array[train_num:],:,:]
    trainset = match_sample[index_rand_array[:train_num]]
    testset=match_sample[index_rand_array[train_num:]]
    # recover_bar=recover_bar[index_rand_array]
    return trainset, testset

def dataset_gen_testset_valid(match_sample,recover_bar,train_por):
    # generate trainset and testset from total dataset
    num=match_sample.shape[0]
    train_num=np.floor(num*train_por).astype('int')

    # randomly choose sample from total data
    index_rand_array = np.arange(match_sample.shape[0])
    np.random.shuffle(index_rand_array)
    trainset = match_sample[index_rand_array[:train_num],:,:]
    testset=match_sample[index_rand_array[train_num:],:,:]
    recover_bar=recover_bar[index_rand_array,:]
    return trainset, testset, recover_bar

def trainset_load():
    # load dataset
    trainset = []
    for i in range(6):
        data = np.load('trainset_recording_' + str(i) + '.npz')
        trainset_recording = data['arr_0']
        if trainset == []:
            trainset = trainset_recording
        else:
            trainset = np.concatenate([trainset, trainset_recording], axis=0)
        print('process_' + str(i))

    train_num = trainset.shape[0]
    index_rand_array = np.arange(trainset.shape[0])
    np.random.shuffle(index_rand_array)
    trainset = trainset[index_rand_array[:train_num], :, :]

    return trainset

'''
def nfold_dataset_gen(div_size,SNR,train_por,mix_por=np.array([1,0,0])):
    # data path
    DataDir_clean_ECG ="/root/autodl-tmp/project/mit_ECG_csv/"
    DataDir_noise_BW = "/root/autodl-tmp/project/mit_noise_csv/"
    DataDir_noise_EM = "/root/autodl-tmp/project/mit_noise_csv/"
    DataDir_noise_MA = "/root/autodl-tmp/project/mit_noise_csv/"

    # DataDir_clean_ECG ="/root/denois_CGAN/mit_ECG_csv/"
    # DataDir_noise_BW = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_EM = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_MA = "/root/denois_CGAN/mit_noise_csv/"

    # noise reading
    Data_noise_BW = pd.read_csv((DataDir_noise_BW+'bw.csv'))
    Data_noise_EM = pd.read_csv((DataDir_noise_EM+'em.csv'))
    Data_noise_MA = pd.read_csv((DataDir_noise_MA+'ma.csv'))
    Data_noise_BW = Data_noise_BW.to_numpy()
    Data_noise_EM = Data_noise_EM.to_numpy()
    Data_noise_MA = Data_noise_MA.to_numpy()

    # clean ECG reading & match data generation
    match_sample=[]
    filenames=os.listdir(DataDir_clean_ECG)
    nfile=len(filenames)

    # random select certain num of files
    # np.random.shuffle(filenames)

    if_train=1
    # trainset gen
    normed_match_sample=0
    if if_train:
        for i in range(int(nfile*train_por)):
            # i=8
            # filenames gen & clean data read
            fulname=(DataDir_clean_ECG+filenames[i])
            print(fulname)
            Data_clean_ECG=pd.read_csv(fulname)
            Data_clean_ECG=Data_clean_ECG.to_numpy()

            # match data generation
            mix_choice=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],
                                     [1,1,1]])
            # mix_choice=np.array([[1,1,1],[1,0,1],[1,1,0],[0,1,1],[1,0,0],[0,0,1],[0,1,0]
            #                          ])
            for train_SNR in range(6):
                # train_SNR=5-_SNR
                for s in range(7):
                    # for b in range(2):
                    #     for e in range(2):
                    #         for m in range(2):
                    # randomly choose a col in noise data as noise
                    col_rand_array = np.arange(Data_noise_BW.shape[1])
                    np.random.shuffle(col_rand_array)
                    # BW = Data_noise_BW[:,col_rand_array[0]]
                    BW = Data_noise_BW[:, 0]  # 1
                    # BW=np.reshape(BW,(Data_clean_ECG.shape[0],1))
                    BW = np.reshape(BW, (BW.shape[0], 1))

                    col_rand_array = np.arange(Data_noise_EM.shape[1])
                    np.random.shuffle(col_rand_array)
                    # EM = Data_noise_EM[:,col_rand_array[0]]
                    EM = Data_noise_EM[:, 0]  # 1
                    # EM=np.reshape(EM,(Data_clean_ECG.shape[0],1))
                    EM = np.reshape(EM, (EM.shape[0], 1))

                    col_rand_array = np.arange(Data_noise_MA.shape[1])
                    np.random.shuffle(col_rand_array)
                    # MA = Data_noise_MA[:,col_rand_array[0]]
                    MA = Data_noise_MA[:, 0]
                    # MA=np.reshape(MA,(Data_clean_ECG.shape[0],1))
                    MA = np.reshape(MA, (MA.shape[0], 1))

                    train_mix_por = mix_choice[s, :]
                    match_sample_long = match_Datagen(Data_clean_ECG, BW, EM, MA,
                                                      train_SNR, train_mix_por)
                    # recording_norm
                    # normed_match_sample_long, normed_recov_bar = recording_norm(match_sample_long)

                    sample = longdata_div(match_sample_long,
                                          div_size)  # match_sample.shape=(sample_num,time,type)
                    # sample = snr_post_data_gen(Data_clean_ECG, BW, EM, MA, SNR, mix_por, div_size)

                    if match_sample == []:
                        match_sample = sample
                    else:
                        match_sample = np.concatenate((match_sample, sample), axis=0)

                        # print('data_num_'+str(i)+str(s)+str(train_SNR)+str(b)+str(e)+str(m))
            # np.savez('trainset_recording_'+str(i)+'.npz', match_sample)
            # print('finished')
            # match_sample_long=match_Datagen(Data_clean_ECG, BW, EM, MA,
            #             SNR, mix_por)
            # sample=longdata_div(match_sample_long,div_size)  # match_sample.shape=(sample_num,time,type)
            # # sample = snr_post_data_gen(Data_clean_ECG, BW, EM, MA, SNR, mix_por, div_size)
            # if match_sample==[]:
            #     match_sample=sample
            # else:
            #     match_sample=np.concatenate((match_sample,sample),axis=0)

    # trainset gen
    # trainset,leftout=dataset_gen(match_sample, train_por=1)
    trainset=0

    # testset gen
    match_sample=[]
    normed_match_sample=0
    for i in range(int(nfile*train_por),int(nfile)):
        # filenames gen & clean data read
        fulname=(DataDir_clean_ECG+filenames[i])
        Data_clean_ECG=pd.read_csv(fulname)
        Data_clean_ECG=Data_clean_ECG.to_numpy()
        print(fulname)
        # randomly choose a col in noise data as noise
        col_rand_array = np.arange(Data_noise_BW.shape[1])
        np.random.shuffle(col_rand_array)
        # BW = Data_noise_BW[:,col_rand_array[0]]
        BW = Data_noise_BW[:, 0] # 1
        # BW=np.reshape(BW,(Data_clean_ECG.shape[0],1))
        BW = np.reshape(BW, (BW.shape[0], 1))

        col_rand_array = np.arange(Data_noise_EM.shape[1])
        np.random.shuffle(col_rand_array)
        # EM = Data_noise_EM[:,col_rand_array[0]]
        EM = Data_noise_EM[:, 0]  # 1
        # EM=np.reshape(EM,(Data_clean_ECG.shape[0],1))
        EM = np.reshape(EM, (EM.shape[0], 1))

        col_rand_array = np.arange(Data_noise_MA.shape[1])
        np.random.shuffle(col_rand_array)
        # MA = Data_noise_MA[:,col_rand_array[0]]
        MA = Data_noise_MA[:, 0]
        # MA=np.reshape(MA,(Data_clean_ECG.shape[0],1))
        MA = np.reshape(MA, (MA.shape[0], 1))

        # match data generation
        match_sample_long=match_Datagen(Data_clean_ECG, BW, EM, MA,
                      SNR, mix_por)
        # recording_norm
        # normed_match_sample_long, normed_recov_bar = recording_norm(match_sample_long)

        sample=longdata_div(match_sample_long,div_size)  # match_sample.shape=(sample_num,time,type)
        # sample = snr_post_data_gen(Data_clean_ECG, BW, EM, MA, SNR, mix_por, div_size)
        if match_sample==[]:
            match_sample=sample
        else:
            match_sample=np.concatenate((match_sample,sample),axis=0)

    # testset gen
    testset,leftout=dataset_gen_testset_valid(match_sample, train_por=1)
    # testset=0
    return trainset,testset
'''

# single train for single noise
def nfold_dataset_gen(div_size,SNR,train_por,mix_por=np.array([1,0,0])):
    # data path
    DataDir_clean_ECG ="/home/uu201912628/project/mit_ECG_csv/"
    DataDir_noise_BW = "/home/uu201912628/project/mit_noise_csv/"
    DataDir_noise_EM = "/home/uu201912628/project/mit_noise_csv/"
    DataDir_noise_MA = "/home/uu201912628/project/mit_noise_csv/"

    # DataDir_clean_ECG ="/root/denois_CGAN/mit_ECG_csv/"
    # DataDir_noise_BW = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_EM = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_MA = "/root/denois_CGAN/mit_noise_csv/"

    # noise reading
    Data_noise_BW = pd.read_csv((DataDir_noise_BW+'bw.csv'))
    Data_noise_EM = pd.read_csv((DataDir_noise_EM+'em.csv'))
    Data_noise_MA = pd.read_csv((DataDir_noise_MA+'ma.csv'))
    Data_noise_BW = Data_noise_BW.to_numpy()
    Data_noise_EM = Data_noise_EM.to_numpy()
    Data_noise_MA = Data_noise_MA.to_numpy()

    # clean ECG reading & match data generation
    match_sample=[]
    recover_bar_total=[]
    filenames=os.listdir(DataDir_clean_ECG)
    nfile=len(filenames)

    # fold control
    name_bank=['selected_ECG_205_MLII.csv','selected_ECG_103_MLII.csv',
               'selected_ECG_219_MLII.csv','selected_ECG_223_MLII.csv',
               'selected_ECG_122_MLII.csv','selected_ECG_111_MLII.csv',
               'selected_ECG_230_MLII.csv','selected_ECG_213_MLII.csv',
               'selected_ECG_116_MLII.csv','selected_ECG_105_MLII.csv']

    # random select certain num of files
    # np.random.shuffle(filenames)

    BW = Data_noise_BW[:, 1]  # 1
    BW = np.reshape(BW, (BW.shape[0], 1))

    EM = Data_noise_EM[:, 1]  # 1
    EM = np.reshape(EM, (EM.shape[0], 1))

    MA = Data_noise_MA[:, 0]
    MA = np.reshape(MA, (MA.shape[0], 1))

    if_train=1
    # trainset gen
    normed_match_sample=0
    snr_bank=np.array([0,1.25,5])
    mix_choice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                           [1, 1, 1]])
    if if_train:
        for i in range(int(nfile*train_por)):
            # filenames gen & clean data read
            fulname = (DataDir_clean_ECG + name_bank[i])
            Data_clean_ECG = pd.read_csv(fulname)
            Data_clean_ECG = Data_clean_ECG.to_numpy()
            print(('train_' + fulname))
            for j in range(snr_bank.shape[0]):
                for k in range(7):
                    SNR = snr_bank[j]
                    # match data generation
                    train_mix_por = mix_choice[k, :]
                    match_sample_long = match_Datagen(Data_clean_ECG, BW, EM, MA,
                                                      SNR, train_mix_por)

                    sample, recover_bar = longdata_div(match_sample_long,
                                                       div_size)  # match_sample.shape=(sample_num,time,type)

                    if match_sample == []:
                        match_sample = sample
                        # recover_bar_total = recover_bar
                    else:
                        match_sample = np.concatenate((match_sample, sample), axis=0)
                        # recover_bar_total = np.concatenate((recover_bar_total, recover_bar), axis=0)

    # trainset gen
    trainset,leftout=dataset_gen(match_sample,recover_bar_total, train_por=1)
    # trainset=0

    # testset gen
    match_sample=[]
    recover_bar_total=[]
    normed_match_sample=0
    # for i in range(int(nfile*train_por),int(nfile)):
    for i in range(int(nfile*train_por),int(nfile)):
        # filenames gen & clean data read
        fulname=(DataDir_clean_ECG+name_bank[i])
        Data_clean_ECG=pd.read_csv(fulname)
        Data_clean_ECG=Data_clean_ECG.to_numpy()
        print(fulname)
        # match data generation
        match_sample_long = match_Datagen(Data_clean_ECG, BW, EM, MA,
                                          SNR, mix_por)

        sample, recover_bar = longdata_div(match_sample_long,
                                           div_size)  # match_sample.shape=(sample_num,time,type)

        if match_sample == []:
            match_sample = sample
            recover_bar_total = recover_bar
        else:
            match_sample = np.concatenate((match_sample, sample), axis=0)
            recover_bar_total = np.concatenate((recover_bar_total, recover_bar), axis=0)


    # testset gen
    testset,leftout,recover_bar_total=dataset_gen_testset_valid(match_sample,recover_bar_total, train_por=1)
    # testset=0
    return trainset,testset,recover_bar_total



def SNR(ground_truth,data):
    noise=data-ground_truth

    snr=10*np.log10((np.sum(ground_truth**2))/(np.sum(noise**2)))
    return snr

def RMSE(ground_truth,data):
    rmse=np.sqrt(np.mean((ground_truth-data)**2))
    return rmse

def baseline_correction(ground_truth,data):
    batch_count=ground_truth.shape[0]
    batch_size=ground_truth.shape[1]
    datadim=ground_truth.shape[2]
    N=512
    data_correction=np.zeros((batch_count,batch_size,datadim,1))
    for j in range(batch_count):
        for i in range(batch_size):
            n = ground_truth[j,i,:,:]-data[j,i,:,:]
            correction = abs(np.fft.fftshift(np.fft.fft(n, N)))[0] * np.ones(
                (n.shape[0]))
            data_correction[j,i,:,:]=data[j,i,:,:]+correction
    return data_correction

def baseline_correction_single_batch(ground_truth,data):
    batch_size=ground_truth.shape[0]
    datadim=ground_truth.shape[1]
    N=512
    data_correction=np.zeros((batch_size,datadim,1))
    for i in range(batch_size):
        n = ground_truth[i,:,:]-data[i,:,:]
        correction = np.fft.fftshift(np.fft.fft(n, N))[0] # * np.ones((n.shape[0]))
        correction=np.reshape(correction,(datadim,1))
        data_correction[i,:,:]=data[i,:,:]+correction
    return data_correction

def piece_CC(ground_truth,data):
    # ground_truth and data.shape=(sample_num,div_size,1)
    ground_truth_long=np.reshape(ground_truth,
                                 (ground_truth.shape[0]*ground_truth.shape[1],1))
    data_long=np.reshape(data,(data.shape[0]*data.shape[1],1))
    cc=long_data_CC(ground_truth_long,data_long)
    return cc
def long_data_CC(ground_truth,data):
    # ground_truth and data.shape=(sample_point_num,1)
    # cor=np.corrcoef(ground_truth,data)
    g_=ground_truth.mean()
    d_=data.mean()

    cc=np.sum((ground_truth-g_)*(data-d_))/\
       ((np.sqrt(np.sum((ground_truth-g_)**2)))*(np.sqrt(np.sum((data-d_)**2))))
    return cc

class SpectralNorm(tf.keras.constraints.Constraint):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter
    def call(self, input_weights):
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
        u = tf.random.normal((w.shape[0], 1))
        v=0
        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a=True)
            v /= tf.norm(v)
            u = tf.matmul(w, v)
            u /= tf.norm(u)
        spec_norm = tf.matmul(u, tf.matmul(w, v),transpose_a=True)
        return input_weights/spec_norm


'''
def create_generator():
    input_layer=keras.Input(shape=(512,1))   # (512,1)
    # generator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))

    # generator encoder
    G_block_1_conv=keras.layers.Conv1D(16,15,strides=1,padding='same')  # 15
    G_block_1_gru=keras.layers.GRU(units=16,return_sequences=True)
    G_block_1_Prelu=keras.layers.PReLU()

    G_block_2_conv=keras.layers.Conv1D(32,15,strides=1,padding='same')   # 15
    G_block_2_gru=keras.layers.GRU(units=32,return_sequences=True)
    G_block_2_Prelu=keras.layers.PReLU()

    G_block_3_conv=keras.layers.Conv1D(64,9,strides=1,padding='same')   # 9
    G_block_3_gru=keras.layers.GRU(units=64,return_sequences=True)
    G_block_3_Prelu=keras.layers.PReLU()

    G_block_4_conv=keras.layers.Conv1D(128,9,strides=2,padding='same')   # 9
    G_block_4_gru=keras.layers.GRU(units=128,return_sequences=True)
    G_block_4_Prelu=keras.layers.PReLU()

    G_block_5_conv=keras.layers.Conv1D(256,5,strides=2,padding='same')   # 5
    G_block_5_gru=keras.layers.GRU(units=256,return_sequences=True)
    G_block_5_Prelu=keras.layers.PReLU()

    G_block_6_conv=keras.layers.Conv1D(512,3,strides=2,padding='same')   # 3
    G_block_6_gru=keras.layers.GRU(units=512,return_sequences=True)
    G_block_6_Prelu=keras.layers.PReLU()

    G_block_7_conv=keras.layers.Conv1D(1024,3,strides=2,padding='same')   # 3
    G_block_7_gru=keras.layers.GRU(units=1024,return_sequences=True)
    G_block_7_Prelu=keras.layers.PReLU()

    # generator decoder
    G_block_8_deconv=keras.layers.Conv1DTranspose(512,5,strides=2,padding='same')   # 5
    G_block_8_gru=keras.layers.GRU(units=512,return_sequences=True)
    G_block_8_Prelu=keras.layers.PReLU()

    G_block_9_deconv=keras.layers.Conv1DTranspose(256,5,strides=2,padding='same')   # 5
    G_block_9_gru=keras.layers.GRU(units=256,return_sequences=True)
    G_block_9_Prelu=keras.layers.PReLU()

    G_block_10_deconv=keras.layers.Conv1DTranspose(128,5,strides=2,padding='same')   # 5
    G_block_10_gru=keras.layers.GRU(units=128,return_sequences=True)
    G_block_10_Prelu=keras.layers.PReLU()

    G_block_11_deconv=keras.layers.Conv1DTranspose(64,3,strides=2,padding='same')   # 3
    G_block_11_gru=keras.layers.GRU(units=64,return_sequences=True)
    G_block_11_Prelu=keras.layers.PReLU()

    G_block_12_deconv=keras.layers.Conv1DTranspose(32,3,strides=1,padding='same')   # 3
    G_block_12_gru=keras.layers.GRU(units=32,return_sequences=True)
    G_block_12_Prelu=keras.layers.PReLU()

    G_block_13_deconv=keras.layers.Conv1DTranspose(16,3,strides=1,padding='same')   # 3
    G_block_13_gru=keras.layers.GRU(units=16,return_sequences=True)
    G_block_13_Prelu=keras.layers.PReLU()

    G_block_14_deconv=keras.layers.Conv1DTranspose(1,3,strides=1,padding='same')   # 3
    G_block_14_gru=keras.layers.GRU(units=1,return_sequences=True)
    G_block_14_Prelu=keras.layers.PReLU()

    # data stream
    layer_1_tensor=G_block_1_conv(input_layer)
    layer_2_tensor=G_block_1_gru(layer_1_tensor)
    layer_3_tensor=G_block_1_Prelu(layer_2_tensor)

    layer_4_tensor=G_block_2_conv(layer_3_tensor)
    layer_5_tensor=G_block_2_gru(layer_4_tensor)
    layer_6_tensor=G_block_2_Prelu(layer_5_tensor)

    layer_7_tensor=G_block_3_conv(layer_6_tensor)
    layer_8_tensor=G_block_3_gru(layer_7_tensor)
    layer_9_tensor=G_block_3_Prelu(layer_8_tensor)

    layer_10_tensor=G_block_4_conv(layer_9_tensor)
    layer_11_tensor=G_block_4_gru(layer_10_tensor)
    layer_12_tensor=G_block_4_Prelu(layer_11_tensor)

    layer_13_tensor=G_block_5_conv(layer_12_tensor)
    layer_14_tensor=G_block_5_gru(layer_13_tensor)
    layer_15_tensor=G_block_5_Prelu(layer_14_tensor)

    layer_16_tensor=G_block_6_conv(layer_15_tensor)
    layer_17_tensor=G_block_6_gru(layer_16_tensor)
    layer_18_tensor=G_block_6_Prelu(layer_17_tensor)

    layer_19_tensor=G_block_7_conv(layer_18_tensor)
    layer_20_tensor=G_block_7_gru(layer_19_tensor)
    through_vector=G_block_7_Prelu(layer_20_tensor)

    # norrand=tf.random.normal(shape=(through_vector.shape[1]))
    # through_vector=tf.concat(through_vector,norrand,axis=2)
    layer_22_tensor=G_block_8_deconv(through_vector)
    layer_23_tensor=G_block_8_gru(layer_22_tensor)
    layer_24_tensor=G_block_8_Prelu(layer_23_tensor+layer_16_tensor)

    layer_25_tensor=G_block_9_deconv(layer_24_tensor)
    layer_26_tensor=G_block_9_gru(layer_25_tensor)
    layer_27_tensor=G_block_9_Prelu(layer_26_tensor+layer_13_tensor)

    layer_28_tensor=G_block_10_deconv(layer_27_tensor)
    layer_29_tensor=G_block_10_gru(layer_28_tensor)
    layer_30_tensor=G_block_10_Prelu(layer_29_tensor+layer_10_tensor)

    layer_31_tensor=G_block_11_deconv(layer_30_tensor)
    layer_32_tensor=G_block_11_gru(layer_31_tensor)
    layer_33_tensor=G_block_11_Prelu(layer_32_tensor+layer_7_tensor)

    layer_34_tensor=G_block_12_deconv(layer_33_tensor)
    layer_35_tensor=G_block_12_gru(layer_34_tensor)
    layer_36_tensor=G_block_12_Prelu(layer_35_tensor+layer_4_tensor)

    layer_37_tensor=G_block_13_deconv(layer_36_tensor)
    layer_38_tensor=G_block_13_gru(layer_37_tensor)
    layer_39_tensor=G_block_13_Prelu(layer_38_tensor+layer_1_tensor)

    layer_40_tensor=G_block_14_deconv(layer_39_tensor)
    layer_41_tensor=G_block_14_gru(layer_40_tensor)
    output_tensor=G_block_14_Prelu(layer_41_tensor)

    # model build
    generator=keras.Model(inputs=input_layer,outputs=output_tensor)

    generator.compile()
    return generator
'''


# gru upgrade
def create_generator():
    input_layer=keras.Input(shape=(1024,12))   # (512,1)
    # generator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))
    # 1024, 1
    # generator encoder
    G_block_0_conv=keras.layers.Conv1D(8,15,strides=1,padding='same')  # 15
    G_block_0_Prelu=keras.layers.PReLU()
    # 512, 8
    G_block_1_conv=keras.layers.Conv1D(16,15,strides=1,padding='same')  # 15
    G_block_1_Prelu=keras.layers.PReLU()
    # 256, 16
    G_block_2_conv=keras.layers.Conv1D(32,15,strides=1,padding='same')   # 15
    G_block_2_Prelu=keras.layers.PReLU()
    # 128, 32
    G_block_3_conv=keras.layers.Conv1D(64,15,strides=2,padding='same')   # 9
    G_block_3_Prelu=keras.layers.PReLU()
    # 64, 64
    G_block_4_conv=keras.layers.Conv1D(128,15,strides=2,padding='same')   # 9
    G_block_4_Prelu=keras.layers.PReLU()
    # 32, 128
    G_block_5_conv=keras.layers.Conv1D(256,15,strides=2,padding='same')   # 5
    G_block_5_Prelu=keras.layers.PReLU()
    # 16,256
    G_block_6_conv=keras.layers.Conv1D(512,15,strides=2,padding='same')   # 3
    G_block_6_Prelu=keras.layers.PReLU()
    # 8, 512
    G_block_7_conv=keras.layers.Conv1D(1024,15,strides=2,padding='same')   # 3
    G_block_7_Prelu=keras.layers.PReLU()
    # 4, 1024
    # generator decoder

    G_block_8_deconv=keras.layers.Conv1DTranspose(512,15,strides=2,padding='same')   # 5
    G_block_8_gru=keras.layers.GRU(units=512,return_sequences=True)
    G_block_8_Prelu=keras.layers.PReLU()
    # 8, 512
    G_block_9_deconv=keras.layers.Conv1DTranspose(256,15,strides=2,padding='same')   # 5
    G_block_9_gru=keras.layers.GRU(units=256,return_sequences=True)
    G_block_9_Prelu=keras.layers.PReLU()
    # 16, 256
    G_block_10_deconv=keras.layers.Conv1DTranspose(128,15,strides=2,padding='same')   # 5
    G_block_10_gru=keras.layers.GRU(units=128,return_sequences=True)
    G_block_10_Prelu=keras.layers.PReLU()
    # 32, 128
    G_block_11_deconv=keras.layers.Conv1DTranspose(64,15,strides=2,padding='same')   # 3
    G_block_11_gru=keras.layers.GRU(units=64,return_sequences=True)
    G_block_11_Prelu=keras.layers.PReLU()
    # 64, 64
    G_block_12_deconv=keras.layers.Conv1DTranspose(32,15,strides=2,padding='same')   # 3
    G_block_12_gru=keras.layers.GRU(units=32,return_sequences=True)
    G_block_12_Prelu=keras.layers.PReLU()
    # 128, 32
    G_block_13_deconv=keras.layers.Conv1DTranspose(16,15,strides=1,padding='same')   # 3
    G_block_13_gru=keras.layers.GRU(units=16,return_sequences=True)
    G_block_13_Prelu=keras.layers.PReLU()
    # 256, 16
    G_block_14_deconv=keras.layers.Conv1DTranspose(8,15,strides=1,padding='same')   # 3
    G_block_14_gru=keras.layers.GRU(units=8,return_sequences=True)
    G_block_14_Prelu=keras.layers.PReLU()
    # 512, 8
    G_block_15_deconv=keras.layers.Conv1DTranspose(1,15,strides=1,padding='same')   # 3
    G_block_15_gru=keras.layers.GRU(units=1,return_sequences=True)
    # G_block_15_Prelu=keras.layers.Activation(tf.nn.tanh)
    # 1024, 1

    # long skip connection
    G_long_skip_conv=keras.layers.Conv1D(filters=32,kernel_size=31,strides=1,padding='same')
    G_long_skip_gru=keras.layers.GRU(units=1,return_sequences=True)

    # data stream
    layer_0_tensor=G_block_0_conv(input_layer)
    # layer_2_tensor=G_block_1_gru(layer_1_tensor)
    layer_0_tensor=G_block_0_Prelu(layer_0_tensor)

    layer_1_tensor=G_block_1_conv(layer_0_tensor)
    # layer_2_tensor=G_block_1_gru(layer_1_tensor)
    layer_3_tensor=G_block_1_Prelu(layer_1_tensor)

    layer_4_tensor=G_block_2_conv(layer_3_tensor)
    # layer_5_tensor=G_block_2_gru(layer_4_tensor)
    layer_6_tensor=G_block_2_Prelu(layer_4_tensor)

    layer_7_tensor=G_block_3_conv(layer_6_tensor)
    # layer_8_tensor=G_block_3_gru(layer_7_tensor)
    layer_9_tensor=G_block_3_Prelu(layer_7_tensor)

    layer_10_tensor=G_block_4_conv(layer_9_tensor)
    # layer_11_tensor=G_block_4_gru(layer_10_tensor)
    layer_12_tensor=G_block_4_Prelu(layer_10_tensor)

    layer_13_tensor=G_block_5_conv(layer_12_tensor)
    # layer_14_tensor=G_block_5_gru(layer_13_tensor)
    layer_15_tensor=G_block_5_Prelu(layer_13_tensor)

    layer_16_tensor=G_block_6_conv(layer_15_tensor)
    # layer_17_tensor=G_block_6_gru(layer_16_tensor)
    layer_18_tensor=G_block_6_Prelu(layer_16_tensor)

    layer_19_tensor=G_block_7_conv(layer_18_tensor)
    # layer_20_tensor=G_block_7_gru(layer_19_tensor)
    through_vector=G_block_7_Prelu(layer_19_tensor)


    layer_22_tensor=G_block_8_deconv(through_vector)
    layer_23_tensor=G_block_8_gru(layer_22_tensor)
    layer_24_tensor=G_block_8_Prelu(layer_23_tensor)

    layer_25_tensor=G_block_9_deconv(tf.concat([layer_24_tensor,layer_18_tensor],axis=2))
    layer_26_tensor=G_block_9_gru(layer_25_tensor)
    layer_27_tensor=G_block_9_Prelu(layer_26_tensor)

    layer_28_tensor=G_block_10_deconv(tf.concat([layer_27_tensor,layer_15_tensor],axis=2))
    layer_29_tensor=G_block_10_gru(layer_28_tensor)
    layer_30_tensor=G_block_10_Prelu(layer_29_tensor)

    layer_31_tensor=G_block_11_deconv(tf.concat([layer_30_tensor,layer_12_tensor],axis=2))
    layer_32_tensor=G_block_11_gru(layer_31_tensor)
    layer_33_tensor=G_block_11_Prelu(layer_32_tensor)

    layer_34_tensor=G_block_12_deconv(tf.concat([layer_33_tensor,layer_9_tensor],axis=2))
    layer_35_tensor=G_block_12_gru(layer_34_tensor)
    layer_36_tensor=G_block_12_Prelu(layer_35_tensor)

    layer_37_tensor=G_block_13_deconv(tf.concat([layer_36_tensor,layer_6_tensor],axis=2))
    # layer_38_tensor=G_block_13_gru(layer_37_tensor)
    layer_39_tensor=G_block_13_Prelu(layer_37_tensor)

    layer_40_tensor=G_block_14_deconv(tf.concat([layer_39_tensor+layer_3_tensor],axis=2))
    # layer_41_tensor=G_block_14_gru(layer_40_tensor)
    layer_42_tensor=G_block_14_Prelu(layer_40_tensor)

    layer_43_tensor=G_block_15_deconv(tf.concat([layer_42_tensor,layer_0_tensor],axis=2))
    layer_44_tensor=G_block_15_gru(layer_43_tensor)
    # layer_44_tensor=G_block_15_Prelu(layer_43_tensor)

    # long skip connection
    # longskip_tensor_1=G_long_skip_conv(input_layer)
    # longskip_tensor_2=G_long_skip_gru(longskip_tensor_1)

    # G tail dense
    # layer_42_tensor=G_dense_1(layer_42_tensor)
    # layer_42_tensor=G_dense_2(layer_42_tensor)
    # layer_42_tensor=G_dense_3(layer_42_tensor)

    output_tensor=layer_44_tensor # +longskip_tensor_2
    # output_tensor=(output_tensor-tf.minimum(output_tensor))/\
    #               (tf.maximum(output_tensor)-tf.minimum(output_tensor))

    # model build
    generator=keras.Model(inputs=input_layer,outputs=output_tensor)

    generator.compile()
    return generator


def create_generator_12lead():
    input_layer=keras.Input(shape=(1024,12))   # (512,1)
    # generator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))
    # 1024, 1
    # generator encoder
    G_block_0_conv=keras.layers.Conv1D(12,15,strides=1,padding='same')  # 15  8
    G_block_0_Prelu=keras.layers.PReLU()
    # 512, 8
    G_block_1_conv=keras.layers.Conv1D(16,15,strides=1,padding='same')  # 15
    G_block_1_Prelu=keras.layers.PReLU()
    # 256, 16
    G_block_2_conv=keras.layers.Conv1D(32,15,strides=1,padding='same')   # 15
    G_block_2_Prelu=keras.layers.PReLU()
    # 128, 32
    G_block_3_conv=keras.layers.Conv1D(64,15,strides=2,padding='same')   # 9
    G_block_3_Prelu=keras.layers.PReLU()
    # 64, 64
    G_block_4_conv=keras.layers.Conv1D(128,15,strides=2,padding='same')   # 9
    G_block_4_Prelu=keras.layers.PReLU()
    # 32, 128
    G_block_5_conv=keras.layers.Conv1D(256,15,strides=2,padding='same')   # 5
    G_block_5_Prelu=keras.layers.PReLU()
    # 16,256
    G_block_6_conv=keras.layers.Conv1D(512,15,strides=2,padding='same')   # 3
    G_block_6_Prelu=keras.layers.PReLU()
    # 8, 512
    G_block_7_conv=keras.layers.Conv1D(1024,15,strides=2,padding='same')   # 3
    G_block_7_Prelu=keras.layers.PReLU()
    # 4, 1024
    # generator decoder

    G_block_8_deconv=keras.layers.Conv1DTranspose(512,15,strides=2,padding='same')   # 5
    G_block_8_gru=keras.layers.GRU(units=512,return_sequences=True)
    G_block_8_Prelu=keras.layers.PReLU()
    # 8, 512
    G_block_9_deconv=keras.layers.Conv1DTranspose(256,15,strides=2,padding='same')   # 5
    G_block_9_gru=keras.layers.GRU(units=256,return_sequences=True)
    G_block_9_Prelu=keras.layers.PReLU()
    # 16, 256
    G_block_10_deconv=keras.layers.Conv1DTranspose(128,15,strides=2,padding='same')   # 5
    G_block_10_gru=keras.layers.GRU(units=128,return_sequences=True)
    G_block_10_Prelu=keras.layers.PReLU()
    # 32, 128
    G_block_11_deconv=keras.layers.Conv1DTranspose(64,15,strides=2,padding='same')   # 3
    G_block_11_gru=keras.layers.GRU(units=64,return_sequences=True)
    G_block_11_Prelu=keras.layers.PReLU()
    # 64, 64
    G_block_12_deconv=keras.layers.Conv1DTranspose(32,15,strides=2,padding='same')   # 3
    G_block_12_gru=keras.layers.GRU(units=32,return_sequences=True)
    G_block_12_Prelu=keras.layers.PReLU()
    # 128, 32
    G_block_13_deconv=keras.layers.Conv1DTranspose(16,15,strides=1,padding='same')   # 3
    G_block_13_gru=keras.layers.GRU(units=16,return_sequences=True)
    G_block_13_Prelu=keras.layers.PReLU()
    # 256, 16
    G_block_14_deconv=keras.layers.Conv1DTranspose(12,15,strides=1,padding='same')   # 3  8
    G_block_14_gru=keras.layers.GRU(units=12,return_sequences=True)
    G_block_14_Prelu=keras.layers.PReLU()
    # 512, 8
    G_block_15_deconv=keras.layers.Conv1DTranspose(12,15,strides=1,padding='same')   # 3
    G_block_15_gru=keras.layers.GRU(units=12,return_sequences=True)
    # G_block_15_Prelu=keras.layers.Activation(tf.nn.tanh)
    # 1024, 1

    # long skip connection
    G_long_skip_conv=keras.layers.Conv1D(filters=32,kernel_size=31,strides=1,padding='same')
    G_long_skip_gru=keras.layers.GRU(units=1,return_sequences=True)

    # data stream
    layer_0_tensor=G_block_0_conv(input_layer)
    # layer_2_tensor=G_block_1_gru(layer_1_tensor)
    layer_0_tensor=G_block_0_Prelu(layer_0_tensor)

    layer_1_tensor=G_block_1_conv(layer_0_tensor)
    # layer_2_tensor=G_block_1_gru(layer_1_tensor)
    layer_3_tensor=G_block_1_Prelu(layer_1_tensor)

    layer_4_tensor=G_block_2_conv(layer_3_tensor)
    # layer_5_tensor=G_block_2_gru(layer_4_tensor)
    layer_6_tensor=G_block_2_Prelu(layer_4_tensor)

    layer_7_tensor=G_block_3_conv(layer_6_tensor)
    # layer_8_tensor=G_block_3_gru(layer_7_tensor)
    layer_9_tensor=G_block_3_Prelu(layer_7_tensor)

    layer_10_tensor=G_block_4_conv(layer_9_tensor)
    # layer_11_tensor=G_block_4_gru(layer_10_tensor)
    layer_12_tensor=G_block_4_Prelu(layer_10_tensor)

    layer_13_tensor=G_block_5_conv(layer_12_tensor)
    # layer_14_tensor=G_block_5_gru(layer_13_tensor)
    layer_15_tensor=G_block_5_Prelu(layer_13_tensor)

    layer_16_tensor=G_block_6_conv(layer_15_tensor)
    # layer_17_tensor=G_block_6_gru(layer_16_tensor)
    layer_18_tensor=G_block_6_Prelu(layer_16_tensor)

    layer_19_tensor=G_block_7_conv(layer_18_tensor)
    # layer_20_tensor=G_block_7_gru(layer_19_tensor)
    through_vector=G_block_7_Prelu(layer_19_tensor)


    layer_22_tensor=G_block_8_deconv(through_vector)
    layer_23_tensor=G_block_8_gru(layer_22_tensor)
    layer_24_tensor=G_block_8_Prelu(layer_23_tensor)

    layer_25_tensor=G_block_9_deconv(tf.concat([layer_24_tensor,layer_18_tensor],axis=2))
    layer_26_tensor=G_block_9_gru(layer_25_tensor)
    layer_27_tensor=G_block_9_Prelu(layer_26_tensor)

    layer_28_tensor=G_block_10_deconv(tf.concat([layer_27_tensor,layer_15_tensor],axis=2))
    layer_29_tensor=G_block_10_gru(layer_28_tensor)
    layer_30_tensor=G_block_10_Prelu(layer_29_tensor)

    layer_31_tensor=G_block_11_deconv(tf.concat([layer_30_tensor,layer_12_tensor],axis=2))
    layer_32_tensor=G_block_11_gru(layer_31_tensor)
    layer_33_tensor=G_block_11_Prelu(layer_32_tensor)

    layer_34_tensor=G_block_12_deconv(tf.concat([layer_33_tensor,layer_9_tensor],axis=2))
    layer_35_tensor=G_block_12_gru(layer_34_tensor)
    layer_36_tensor=G_block_12_Prelu(layer_35_tensor)

    layer_37_tensor=G_block_13_deconv(tf.concat([layer_36_tensor,layer_6_tensor],axis=2))
    # layer_38_tensor=G_block_13_gru(layer_37_tensor)
    layer_39_tensor=G_block_13_Prelu(layer_37_tensor)

    layer_40_tensor=G_block_14_deconv(tf.concat([layer_39_tensor+layer_3_tensor],axis=2))
    # layer_41_tensor=G_block_14_gru(layer_40_tensor)
    layer_42_tensor=G_block_14_Prelu(layer_40_tensor)

    layer_43_tensor=G_block_15_deconv(tf.concat([layer_42_tensor,layer_0_tensor],axis=2))
    layer_44_tensor=G_block_15_gru(layer_43_tensor)
    # layer_44_tensor=G_block_15_Prelu(layer_43_tensor)

    # long skip connection
    # longskip_tensor_1=G_long_skip_conv(input_layer)
    # longskip_tensor_2=G_long_skip_gru(longskip_tensor_1)

    # G tail dense
    # layer_42_tensor=G_dense_1(layer_42_tensor)
    # layer_42_tensor=G_dense_2(layer_42_tensor)
    # layer_42_tensor=G_dense_3(layer_42_tensor)

    output_tensor=layer_44_tensor # +longskip_tensor_2
    # output_tensor=(output_tensor-tf.minimum(output_tensor))/\
    #               (tf.maximum(output_tensor)-tf.minimum(output_tensor))

    # model build
    generator=keras.Model(inputs=input_layer,outputs=output_tensor)

    generator.compile()
    return generator


'''
# before
def create_generator():
    input_layer=keras.Input(shape=(512,1))   # (512,1)
    # generator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))

    # generator encoder
    G_layer_1_conv=keras.layers.Conv1D(16,9,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')  # 9
    G_layer_2_Prelu=keras.layers.PReLU()

    G_layer_3_conv=keras.layers.Conv1D(32,7,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')   # 7
    G_layer_4_Prelu=keras.layers.PReLU()

    G_layer_5_conv=keras.layers.Conv1D(64,7,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')   # 7
    G_layer_6_Prelu=keras.layers.PReLU()

    G_layer_7_conv=keras.layers.Conv1D(128,5,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')   # 5
    G_layer_8_Prelu=keras.layers.PReLU()

    G_layer_9_conv=keras.layers.Conv1D(256,5,kernel_constraint = SpectralNorm(),
                                       strides=1,padding='same')   # 5
    G_layer_10_Prelu=keras.layers.PReLU()

    G_layer_11_conv=keras.layers.Conv1D(512,3,kernel_constraint = SpectralNorm(),
                                        strides=1,padding='same')   # 3
    G_layer_12_Prelu=keras.layers.PReLU()

    G_layer_13_conv=keras.layers.Conv1D(1024,3,kernel_constraint = SpectralNorm(),
                                        strides=1,padding='same')   # 3
    G_layer_14_Prelu=keras.layers.PReLU()

    # through
    # G_select_dense_1=keras.layers.Dense(1024,activation='tanh')
    # G_select_acti_1=keras.layers.PReLU()
    #
    # G_select_dense_2=keras.layers.Dense(1024,activation='sigmoid')
    # G_select_acti_2=keras.layers.PReLU()

    # generator decoder
    G_layer_15_deconv=keras.layers.Conv1DTranspose(512,7,kernel_constraint = SpectralNorm(),
                                                   strides=1,padding='same')   # 7
    G_layer_16_Prelu=keras.layers.PReLU()

    G_layer_17_deconv=keras.layers.Conv1DTranspose(256,5,kernel_constraint = SpectralNorm(),
                                                   strides=1,padding='same')   # 5
    G_layer_18_Prelu=keras.layers.PReLU()

    G_layer_19_deconv=keras.layers.Conv1DTranspose(128,5,kernel_constraint = SpectralNorm(),
                                                   strides=1,padding='same')   # 5
    G_layer_20_Prelu=keras.layers.PReLU()

    G_layer_21_deconv=keras.layers.Conv1DTranspose(64,3,kernel_constraint = SpectralNorm(),
                                                   strides=2,padding='same')   # 3
    G_layer_22_Prelu=keras.layers.PReLU()

    G_layer_23_deconv=keras.layers.Conv1DTranspose(32,3,kernel_constraint = SpectralNorm(),
                                                   strides=2,padding='same')   # 3
    G_layer_24_Prelu=keras.layers.PReLU()

    G_layer_25_deconv=keras.layers.Conv1DTranspose(16,3,kernel_constraint = SpectralNorm(),
                                                   strides=2,padding='same')   # 3
    G_layer_26_Prelu=keras.layers.PReLU()

    G_layer_27_deconv=keras.layers.Conv1DTranspose(1,3,kernel_constraint = SpectralNorm(),
                                                   strides=2,padding='same')   # 3
    G_layer_28_Prelu=keras.layers.PReLU()

    # data stream
    layer_1_tensor=G_layer_1_conv(input_layer)
    layer_2_tensor=G_layer_2_Prelu(layer_1_tensor)
    layer_3_tensor=G_layer_3_conv(layer_2_tensor)
    layer_4_tensor=G_layer_4_Prelu(layer_3_tensor)
    layer_5_tensor=G_layer_5_conv(layer_4_tensor)
    layer_6_tensor=G_layer_6_Prelu(layer_5_tensor)
    layer_7_tensor=G_layer_7_conv(layer_6_tensor)
    layer_8_tensor=G_layer_8_Prelu(layer_7_tensor)
    layer_9_tensor=G_layer_9_conv(layer_8_tensor)
    layer_10_tensor=G_layer_10_Prelu(layer_9_tensor)
    layer_11_tensor=G_layer_11_conv(layer_10_tensor)
    layer_12_tensor=G_layer_12_Prelu(layer_11_tensor)
    layer_13_tensor=G_layer_13_conv(layer_12_tensor)
    layer_14_tensor=G_layer_14_Prelu(layer_13_tensor)

    through_vector=layer_14_tensor
    # through_vector=G_select_dense_1(through_vector)
    # through_vector=G_select_acti_1(through_vector)
    # through_vector=G_select_dense_2(through_vector)
    # through_vector=G_select_acti_2(through_vector)

    layer_15_tensor=G_layer_15_deconv(through_vector)
    layer_16_tensor=G_layer_16_Prelu(layer_15_tensor+layer_11_tensor)
    layer_17_tensor=G_layer_17_deconv(layer_16_tensor)
    layer_18_tensor=G_layer_18_Prelu(layer_17_tensor+layer_9_tensor)
    layer_19_tensor=G_layer_19_deconv(layer_18_tensor)
    layer_20_tensor=G_layer_20_Prelu(layer_19_tensor+layer_7_tensor)
    layer_21_tensor=G_layer_21_deconv(layer_20_tensor)
    layer_22_tensor=G_layer_22_Prelu(layer_21_tensor+layer_5_tensor)
    layer_23_tensor=G_layer_23_deconv(layer_22_tensor)
    layer_24_tensor=G_layer_24_Prelu(layer_23_tensor+layer_3_tensor)
    layer_25_tensor=G_layer_25_deconv(layer_24_tensor)
    layer_26_tensor=G_layer_26_Prelu(layer_25_tensor+layer_1_tensor)
    layer_27_tensor=G_layer_27_deconv(layer_26_tensor)
    output_tensor=G_layer_28_Prelu(layer_27_tensor)

    # model build
    generator=keras.Model(inputs=input_layer,outputs=output_tensor)

    generator.compile()
    generator.summary()
    return generator
'''

'''
# conv+upsampling
def create_generator():
    input_layer=keras.Input(shape=(512,1))   # (512,1)
    # generator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))

    # generator encoder
    G_layer_1_conv=keras.layers.Conv1D(16,9,strides=2,padding='same')  # 9
    G_layer_2_Prelu=keras.layers.PReLU()

    G_layer_3_conv=keras.layers.Conv1D(32,7,strides=2,padding='same')   # 7
    G_layer_4_Prelu=keras.layers.PReLU()

    G_layer_5_conv=keras.layers.Conv1D(64,7,strides=2,padding='same')   # 7
    G_layer_6_Prelu=keras.layers.PReLU()

    G_layer_7_conv=keras.layers.Conv1D(128,5,strides=2,padding='same')   # 5
    G_layer_8_Prelu=keras.layers.PReLU()

    G_layer_9_conv=keras.layers.Conv1D(256,5,strides=2,padding='same')   # 5
    G_layer_10_Prelu=keras.layers.PReLU()

    G_layer_11_conv=keras.layers.Conv1D(512,3,strides=2,padding='same')   # 3
    G_layer_12_Prelu=keras.layers.PReLU()

    G_layer_13_conv=keras.layers.Conv1D(1024,3,strides=2,padding='same')   # 3
    G_layer_14_Prelu=keras.layers.PReLU()

    # generator decoder
    G_layer_15_deconv=keras.layers.Conv1D(512,7,strides=1,padding='same')   # 7
    G_layer_16_Prelu=keras.layers.UpSampling1D(2)

    G_layer_17_deconv=keras.layers.Conv1D(256,5,strides=1,padding='same')   # 5
    G_layer_18_Prelu=keras.layers.UpSampling1D(2)

    G_layer_19_deconv=keras.layers.Conv1D(128,5,strides=1,padding='same')   # 5
    G_layer_20_Prelu=keras.layers.UpSampling1D(2)

    G_layer_21_deconv=keras.layers.Conv1D(64,3,strides=1,padding='same')   # 3
    G_layer_22_Prelu=keras.layers.UpSampling1D(2)

    G_layer_23_deconv=keras.layers.Conv1D(32,3,strides=1,padding='same')   # 3
    G_layer_24_Prelu=keras.layers.UpSampling1D(2)

    G_layer_25_deconv=keras.layers.Conv1D(16,3,strides=1,padding='same')   # 3
    G_layer_26_Prelu=keras.layers.UpSampling1D(2)

    G_layer_27_deconv=keras.layers.Conv1D(1,3,strides=1,padding='same')   # 3
    G_layer_28_Prelu=keras.layers.UpSampling1D(2)

    # data stream
    layer_1_tensor=G_layer_1_conv(input_layer)
    layer_2_tensor=G_layer_2_Prelu(layer_1_tensor)
    layer_3_tensor=G_layer_3_conv(layer_2_tensor)
    layer_4_tensor=G_layer_4_Prelu(layer_3_tensor)
    layer_5_tensor=G_layer_5_conv(layer_4_tensor)
    layer_6_tensor=G_layer_6_Prelu(layer_5_tensor)
    layer_7_tensor=G_layer_7_conv(layer_6_tensor)
    layer_8_tensor=G_layer_8_Prelu(layer_7_tensor)
    layer_9_tensor=G_layer_9_conv(layer_8_tensor)
    layer_10_tensor=G_layer_10_Prelu(layer_9_tensor)
    layer_11_tensor=G_layer_11_conv(layer_10_tensor)
    layer_12_tensor=G_layer_12_Prelu(layer_11_tensor)
    layer_13_tensor=G_layer_13_conv(layer_12_tensor)
    layer_14_tensor=G_layer_14_Prelu(layer_13_tensor)

    layer_15_tensor=G_layer_15_deconv(layer_14_tensor)
    layer_16_tensor=G_layer_16_Prelu(layer_15_tensor+layer_11_tensor)
    layer_17_tensor=G_layer_17_deconv(layer_16_tensor)
    layer_18_tensor=G_layer_18_Prelu(layer_17_tensor)
    layer_19_tensor=G_layer_19_deconv(layer_18_tensor)
    layer_20_tensor=G_layer_20_Prelu(layer_19_tensor)
    layer_21_tensor=G_layer_21_deconv(layer_20_tensor)
    layer_22_tensor=G_layer_22_Prelu(layer_21_tensor)
    layer_23_tensor=G_layer_23_deconv(layer_22_tensor)
    layer_24_tensor=G_layer_24_Prelu(layer_23_tensor)
    layer_25_tensor=G_layer_25_deconv(layer_24_tensor)
    layer_26_tensor=G_layer_26_Prelu(layer_25_tensor)
    layer_27_tensor=G_layer_27_deconv(layer_26_tensor)
    output_tensor=G_layer_28_Prelu(layer_27_tensor)

    # model build
    generator=keras.Model(inputs=input_layer,outputs=output_tensor)

    generator.compile()
    generator.summary()
    return generator
'''

'''
# try
def create_generator():
    discriminator = Sequential()
    discriminator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))  #(512,2)

    # discriminator.add(keras.layers.Dense(310))
    # # discriminator.add(keras.layers.PReLU())
    discriminator.add(keras.layers.Conv1D(64,11,strides=1,padding='same'))
    # discriminator.add(keras.layers.LSTM(64, return_sequences='true'))
    discriminator.add(keras.layers.PReLU())
    
    discriminator.add(keras.layers.Conv1D(128,7,strides=1,padding='same'))
    discriminator.add(keras.layers.PReLU())
    
    # discriminator.add(keras.layers.Conv1D(512,7,activation='tanh',strides=1,padding='same'))
    # discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Dense(250,activation='tanh'))
    # discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Dense(250,activation='tanh'))
    # discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Dense(250,activation='sigmoid'))
    # discriminator.add(keras.layers.PReLU())

    # discriminator.add(keras.layers.Dense(310))
    # # discriminator.add(keras.layers.PReLU())
    discriminator.add(keras.layers.Dense(1))
    discriminator.compile()
    return discriminator
'''

'''
# over fit 
def create_generator():
    input_layer=keras.Input(shape=(512,1))   # (512,1)
    # generator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))

    # generator encoder
    G_layer_1_conv=keras.layers.Conv1D(16,33,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')  # 9
    # G_layer_1_norm=keras.layers.BatchNormalization()
    G_layer_2_Prelu=keras.layers.LeakyReLU()

    G_layer_3_conv=keras.layers.Conv1D(32,33,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')   # 7
    # G_layer_3_norm=keras.layers.BatchNormalization()
    G_layer_4_Prelu=keras.layers.LeakyReLU()

    G_layer_5_conv=keras.layers.Conv1D(64,33,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')   # 7
    # G_layer_5_norm=keras.layers.BatchNormalization()
    G_layer_6_Prelu=keras.layers.LeakyReLU()

    G_layer_7_conv=keras.layers.Conv1D(128,33,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')   # 5
    # G_layer_7_norm=keras.layers.BatchNormalization()
    G_layer_8_Prelu=keras.layers.LeakyReLU()

    G_layer_9_conv=keras.layers.Conv1D(256,33,kernel_constraint = SpectralNorm(),
                                       strides=1,padding='same')   # 5
    # G_layer_9_norm=keras.layers.BatchNormalization()
    G_layer_10_Prelu=keras.layers.LeakyReLU()

    G_layer_11_conv=keras.layers.Conv1D(512,33,kernel_constraint = SpectralNorm(),
                                        strides=1,padding='same')   # 3
    G_layer_12_Prelu=keras.layers.LeakyReLU()

    G_layer_13_conv=keras.layers.Conv1D(1024,33,kernel_constraint = SpectralNorm(),
                                        strides=1,padding='same')   # 3
    G_layer_14_Prelu=keras.layers.LeakyReLU()

    # generator decoder
    G_layer_15_deconv=keras.layers.Dense(512,kernel_regularizer='l2')   # 7
    G_layer_15_up=keras.layers.UpSampling1D(1)
    G_layer_16_Prelu=keras.layers.PReLU()
    # G_layer_16_Prelu = keras.layers.Activation(tf.nn.tanh)

    G_layer_17_deconv=keras.layers.Dense(256,kernel_regularizer='l2')   # 5
    G_layer_17_up=keras.layers.UpSampling1D(1)
    # G_layer_17_norm=keras.layers.BatchNormalization(virtual_batch_size=64)
    G_layer_18_Prelu=keras.layers.PReLU()
    # G_layer_18_Prelu = keras.layers.Activation(tf.nn.tanh)

    G_layer_19_deconv=keras.layers.Dense(128,kernel_regularizer='l2')   # 5
    G_layer_19_up=keras.layers.UpSampling1D(1)
    # G_layer_19_norm=keras.layers.BatchNormalization(virtual_batch_size=64)
    G_layer_20_Prelu=keras.layers.PReLU()
    # G_layer_20_Prelu = keras.layers.Activation(tf.nn.tanh)

    G_layer_21_deconv=keras.layers.Dense(64,kernel_regularizer='l2')   # 3
    G_layer_21_up=keras.layers.UpSampling1D(2)
    # G_layer_21_norm=keras.layers.BatchNormalization(virtual_batch_size=64)
    G_layer_22_Prelu=keras.layers.PReLU()
    # G_layer_22_Prelu = keras.layers.Activation(tf.nn.tanh)

    G_layer_23_deconv=keras.layers.Dense(32,kernel_regularizer='l2')   # 3
    G_layer_23_up=keras.layers.UpSampling1D(2)
    # G_layer_24_Prelu = keras.layers.Activation(tf.nn.tanh)
    G_layer_24_Prelu=keras.layers.PReLU()

    G_layer_25_deconv=keras.layers.Dense(16,kernel_regularizer='l2')   # 3
    G_layer_25_up=keras.layers.UpSampling1D(2)
    # G_layer_26_Prelu = keras.layers.Activation(tf.nn.tanh)
    G_layer_26_Prelu=keras.layers.PReLU()

    G_layer_27_deconv=keras.layers.Dense(1)   # 3
    G_layer_27_up=keras.layers.UpSampling1D(2)
    # G_layer_28_Prelu = keras.layers.PReLU()
    G_layer_28_Prelu = keras.layers.Activation(tf.nn.tanh)
    # G_layer_28_Prelu = keras.layers.PReLU()

    # data stream
    layer_1_tensor=G_layer_1_conv(input_layer)
    # layer_1_tensor=G_layer_1_norm(layer_1_tensor)
    layer_2_tensor=G_layer_2_Prelu(layer_1_tensor)
    layer_3_tensor=G_layer_3_conv(layer_2_tensor)
    # layer_3_tensor=G_layer_3_norm(layer_3_tensor)
    layer_4_tensor=G_layer_4_Prelu(layer_3_tensor)
    layer_5_tensor=G_layer_5_conv(layer_4_tensor)
    # layer_5_tensor=G_layer_5_norm(layer_5_tensor)
    layer_6_tensor=G_layer_6_Prelu(layer_5_tensor)
    layer_7_tensor=G_layer_7_conv(layer_6_tensor)
    # layer_7_tensor=G_layer_7_norm(layer_7_tensor)
    layer_8_tensor=G_layer_8_Prelu(layer_7_tensor)
    layer_9_tensor=G_layer_9_conv(layer_8_tensor)
    # layer_9_tensor=G_layer_9_norm(layer_9_tensor)
    layer_10_tensor=G_layer_10_Prelu(layer_9_tensor)
    layer_11_tensor=G_layer_11_conv(layer_10_tensor)
    layer_12_tensor=G_layer_12_Prelu(layer_11_tensor)
    layer_13_tensor=G_layer_13_conv(layer_12_tensor)
    layer_14_tensor=G_layer_14_Prelu(layer_13_tensor)

    layer_15_tensor=G_layer_15_deconv(layer_14_tensor)
    layer_15_tensor=G_layer_15_up(layer_15_tensor)
    # layer_15_tensor=G_layer_15_norm(layer_15_tensor)
    layer_16_tensor=G_layer_16_Prelu(layer_15_tensor+layer_11_tensor)
    layer_17_tensor=G_layer_17_deconv(layer_16_tensor)
    layer_17_tensor=G_layer_17_up(layer_17_tensor)
    # layer_17_tensor=G_layer_17_norm(layer_17_tensor)
    layer_18_tensor=G_layer_18_Prelu(layer_17_tensor+layer_9_tensor)
    layer_19_tensor=G_layer_19_deconv(layer_18_tensor)
    layer_19_tensor=G_layer_19_up(layer_19_tensor)
    # layer_19_tensor=G_layer_19_norm(layer_19_tensor)
    layer_20_tensor=G_layer_20_Prelu(layer_19_tensor+layer_7_tensor)
    layer_21_tensor=G_layer_21_deconv(layer_20_tensor)
    layer_21_tensor=G_layer_21_up(layer_21_tensor)
    # layer_21_tensor=G_layer_21_norm(layer_21_tensor)
    layer_22_tensor=G_layer_22_Prelu(layer_21_tensor+layer_5_tensor)
    layer_23_tensor=G_layer_23_deconv(layer_22_tensor)
    layer_23_tensor=G_layer_23_up(layer_23_tensor)
    layer_24_tensor=G_layer_24_Prelu(layer_23_tensor+layer_3_tensor)
    layer_25_tensor=G_layer_25_deconv(layer_24_tensor)
    layer_25_tensor=G_layer_25_up(layer_25_tensor)
    layer_26_tensor=G_layer_26_Prelu(layer_25_tensor+layer_1_tensor)
    layer_27_tensor=G_layer_27_deconv(layer_26_tensor)
    layer_27_tensor=G_layer_27_up(layer_27_tensor)
    output_tensor=G_layer_28_Prelu(layer_27_tensor)
    # output_tensor=layer_27_tensor

    # model build
    generator=keras.Model(inputs=input_layer,outputs=output_tensor)

    generator.compile()
    return generator
'''

'''
def create_generator():
    input_layer=keras.Input(shape=(512,1))   # (512,1)
    # generator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))

    # generator encoder
    G_layer_1_conv=keras.layers.Conv1D(16,33,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')  # 9
    # G_layer_1_norm=keras.layers.BatchNormalization()
    G_layer_2_Prelu=keras.layers.LeakyReLU()

    G_layer_3_conv=keras.layers.Conv1D(32,33,kernel_constraint = SpectralNorm(),
                                       strides=2,padding='same')   # 7
    # G_layer_3_norm=keras.layers.BatchNormalization()
    G_layer_4_Prelu=keras.layers.LeakyReLU()


    # generator decoder
    G_layer_25_deconv=keras.layers.Dense(16,kernel_regularizer='l2')   # 3
    G_layer_25_up=keras.layers.UpSampling1D(2)
    # G_layer_26_Prelu = keras.layers.Activation(tf.nn.tanh)
    G_layer_26_Prelu=keras.layers.PReLU()

    G_layer_27_deconv=keras.layers.Dense(1)   # 3
    G_layer_27_up=keras.layers.UpSampling1D(2)
    G_layer_28_Prelu = keras.layers.Activation(tf.nn.tanh)
    # G_layer_28_Prelu = keras.layers.PReLU()

    # data stream
    layer_1_tensor=G_layer_1_conv(input_layer)
    # layer_1_tensor=G_layer_1_norm(layer_1_tensor)
    layer_2_tensor=G_layer_2_Prelu(layer_1_tensor)
    layer_3_tensor=G_layer_3_conv(layer_2_tensor)
    # layer_3_tensor=G_layer_3_norm(layer_3_tensor)
    layer_4_tensor=G_layer_4_Prelu(layer_3_tensor)

    layer_25_tensor=G_layer_25_deconv(layer_4_tensor)
    layer_25_tensor=G_layer_25_up(layer_25_tensor)
    layer_26_tensor=G_layer_26_Prelu(layer_25_tensor+layer_1_tensor)
    layer_27_tensor=G_layer_27_deconv(layer_26_tensor)
    layer_27_tensor=G_layer_27_up(layer_27_tensor)
    output_tensor=G_layer_28_Prelu(layer_27_tensor)

    # model build
    generator=keras.Model(inputs=input_layer,outputs=output_tensor)

    generator.compile()
    return generator
'''

'''
def create_generator():
    model = Sequential()
    model.add(Conv1D(128, 55, activation='relu',padding='same', input_shape=(1024, 1)))
    # model.add(MaxPooling1D(10))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 25, activation='relu',padding='same'))
    # model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 10, activation='relu',padding='same'))
    # model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 5, activation='relu',padding='same'))
    # model.add(GlobalAveragePooling1D())
    # model.add(Flatten())
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile()
    return model
'''
'''
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(tf.keras.layers.InputLayer(input_shape=(1024,2)))  #(512,2)

    discriminator.add(keras.layers.Conv1D(16,31,strides=2,padding='same'))
    discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Conv1D(32,31,strides=2,padding='same'))
    discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Conv1D(64,31,strides=2,padding='same'))
    discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Conv1D(128,31,strides=2,padding='same'))
    discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Conv1D(256,31,strides=2,padding='same'))
    discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Conv1D(512,31,strides=2,padding='same'))
    discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Conv1D(1024,31,strides=2,padding='same'))
    discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Conv1D(1,1,strides=1,padding='same'))
    discriminator.add(keras.layers.Flatten())

    discriminator.add(keras.layers.Dense(units=1, activation='sigmoid'))

    discriminator.compile()
    return discriminator
'''


# over fit
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(tf.keras.layers.InputLayer(input_shape=(1024,12)))  #(512,2)

    discriminator.add(keras.layers.Conv1D(1024,15,strides=2))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(tf.keras.layers.LeakyReLU())

    discriminator.add(keras.layers.Conv1D(512,15,strides=2)) # 512,5
    discriminator.add(keras.layers.Dropout(0.1))  # 0.3
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(tf.keras.layers.LeakyReLU())

    discriminator.add(keras.layers.Conv1D(128,15,strides=2)) # 512,3
    # discriminator.add(keras.layers.Dense(128,activation='tanh'))
    # discriminator.add(keras.layers.Dense(128,activation='sigmoid'))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(tf.keras.layers.LeakyReLU())

    discriminator.add(keras.layers.Conv1D(64,15,strides=2))
    # discriminator.add(keras.layers.Dense(64,activation='tanh'))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(tf.keras.layers.LeakyReLU())

    discriminator.add(keras.layers.Conv1D(32,15,strides=2))
    # discriminator.add(keras.layers.Dense(64,activation='tanh'))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(tf.keras.layers.LeakyReLU())

    discriminator.add(keras.layers.Flatten())
    # discriminator.add(keras.layers.Dense(32))
    # discriminator.add(keras.layers.PReLU())
    # discriminator.add(keras.layers.Dense(16))
    discriminator.add(keras.layers.Dense(units=1, activation='sigmoid'))

    discriminator.compile()
    return discriminator

'''
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(tf.keras.layers.InputLayer(input_shape=(512,1)))  #(512,2)

    # discriminator.add(keras.layers.Conv1D(512,5,strides=2))
    # discriminator.add(keras.layers.BatchNormalization())
    # discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Conv1D(128,3,strides=2))
    # discriminator.add(keras.layers.Dense(64,activation='tanh'))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.PReLU())

    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(32))
    discriminator.add(keras.layers.PReLU())
    discriminator.add(keras.layers.Dense(16))
    discriminator.add(keras.layers.Dense(units=1, activation='sigmoid'))

    discriminator.compile()
    return discriminator
'''

def plot_epoch_denoised_signals(batched_trainset_noisy, batched_trainset_clean, batched_testset_noisy,
                                batched_testset_clean, epoch, generator,batch_count):
    # rand=int(np.random.randint(0,batch_count,1))
    rand=2
    batch_size=batched_trainset_clean.shape[1]
    datadim=batched_trainset_clean.shape[2]
    channel=batched_trainset_clean.shape[3]
    noisy_input = batched_trainset_noisy
    ground_truth=batched_trainset_clean

    test_ground_truth=batched_testset_clean
    test_noisy_input=batched_testset_noisy
    # min=normed_recov_bar[rand,0]
    # max=normed_recov_bar[rand,1]

    # noisy_input=np.reshape(noisy_input,(batch_size,datadim,channel-1))
    # ground_truth=np.reshape(ground_truth,(batch_size,datadim,1))
    # test_ground_truth=np.reshape(test_ground_truth,(batch_size,datadim,1))
    # test_noisy_input=np.reshape(test_noisy_input,(batch_size,datadim,channel-1))

    denoised_signal = generator.predict(noisy_input[0])
    # denoised_signal=baseline_correction_single_batch(ground_truth,denoised_signal)

    # for j in range(batch_size):
    #     denoised_signal[j,:,:]=(denoised_signal[j,:,:]-np.min(denoised_signal[j,:,:]))/\
    #                            (np.max(denoised_signal[j,:,:])-np.min(denoised_signal[j,:,:]))
    # validation
    test_denoise_signal=generator.predict(test_noisy_input)
    # test_denoise_signal=baseline_correction_single_batch(test_ground_truth, test_denoise_signal)
    # for j in range(batch_size):
    #     test_denoise_signal[j,:,:]=(test_denoise_signal[j,:,:]-np.min(test_denoise_signal[j,:,:]))/\
    #                            (np.max(test_denoise_signal[j,:,:])-np.min(test_denoise_signal[j,:,:]))


    rand2=8
    # select a piece to show
    ground_truth_plot = ground_truth[rand2,0,:,0]
    noisy_input_plot=noisy_input[rand2,0,:,0]
    denoised_signal_plot=denoised_signal[0,:,0]

    test_ground_truth_plot=test_ground_truth[0,0,:,1]
    test_noisy_input_plot=test_noisy_input[0,0,:,1]
    test_denoise_signal_plot=test_denoise_signal[0,:,1]

    # performance evaluation
    train_snr=SNR(ground_truth,denoised_signal)
    train_rmse=RMSE(ground_truth,denoised_signal)
    valid_snr=SNR(test_ground_truth,test_denoise_signal)
    valid_rmse=RMSE(test_ground_truth,test_denoise_signal)

    # result=np.asarray([train_snr,train_rmse])
    # pd.DataFrame(result).to_csv(('/home/zhanghaowei/project/denoiseCGAN/train_process/train_process_'+str(epoch)+'.csv'))
    # pd.DataFrame(result).to_csv(
    #     ('/home/uu201912628/project/train_process/train_process_' + str(epoch) + '.csv'))
    # pd.DataFrame(result).to_csv(
    #     ('/root/denois_CGAN/train_process/train_process_' + str(epoch) + '.csv'))

    print("epoch",epoch," training, snr=",train_snr," rmse=",train_rmse)
    print("epoch", epoch, " validing, snr=", valid_snr, " rmse=", valid_rmse)

    N=1536
    # ground_truth_plot_baseline=np.abs(np.fft.fft(ground_truth_plot,axis=0))[0]*np.ones((ground_truth_plot.shape[0]))
    # denoised_signal_plot_baseline=np.abs(np.fft.fft(denoised_signal_plot,axis=0))[0,0]*np.ones((denoised_signal_plot.shape[0]))
    # noisy_input_plot_baseline=np.abs(np.fft.fft(noisy_input_plot,axis=0))[0]*np.ones((noisy_input_plot.shape[0]))

    # test_ground_truth_plot_baseline=np.abs(np.fft.fft(test_ground_truth_plot,axis=0))[0]*np.ones((test_ground_truth_plot.shape[0]))
    # test_denoise_signal_plot_baseline=np.abs(np.fft.fft(test_denoise_signal_plot,axis=0))[0,0]*np.ones((test_denoise_signal_plot.shape[0]))
    # test_noisy_input_plot_baseline=np.abs(np.fft.fft(test_noisy_input_plot,axis=0))[0]*np.ones((test_noisy_input_plot.shape[0]))

    print(("epoch "+str(epoch)+"training denoise plot"))
    plt.subplot(221)
    x=range(len(ground_truth_plot))
    plt.plot(x,ground_truth_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('ground truth')
    plt.subplot(222)
    plt.plot(x,noisy_input_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('noisy signal')
    plt.subplot(223)
    x=range(len(denoised_signal_plot))
    plt.plot(x,denoised_signal_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('denoise signal')
    plt.subplot(224)
    x=range(len(ground_truth_plot))
    plt.plot(x,ground_truth_plot,x,denoised_signal_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('compare of ground truth and denoise signal')
    plt.tight_layout()
    # muti GPU version
    # plt.savefig(('/home/zhanghaowei/project/denoiseCGAN/train_fig/fig-{}.png'.format(epoch)))
    # plt.savefig(('/home/uu201912628/project/train_fig/train_fig-{}.png'.format(epoch)))
    # plt.savefig(('/root/denois_CGAN/train_fig/fig-{}.png'.format(epoch)))
    plt.show()
    plt.close()

    # valid plot
    print(("epoch "+str(epoch)+"validing denoise plot"))
    plt.subplot(221)
    x=range(len(test_ground_truth_plot))
    plt.plot(x,test_ground_truth_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('ground truth')
    plt.subplot(222)
    plt.plot(x,test_noisy_input_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('noisy signal')
    plt.subplot(223)
    x=range(len(test_denoise_signal_plot))
    plt.plot(x,test_denoise_signal_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('denoise signal')
    plt.subplot(224)
    x=range(len(test_ground_truth_plot))
    plt.plot(x,test_ground_truth_plot,x,test_denoise_signal_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('valid')
    plt.tight_layout()
    # muti GPU version
    # plt.savefig(('/home/zhanghaowei/project/denoiseCGAN/train_fig/fig-{}.png'.format(epoch)))
    # plt.savefig(('/home/uu201912628/project/train_fig/valid_fig-{}.png'.format(epoch)))
    # plt.savefig(('/root/denois_CGAN/train_fig/fig-{}.png'.format(epoch)))
    plt.show()
    plt.close()

    return train_snr


class CGAN(Model):
    def __init__(self):
        super(CGAN, self).__init__()

        self.layer_1 = create_generator_12lead()
        self.layer_2 = create_discriminator()

    def G_loss(self, inputs, raw_data,epoch):
        g_output = self.layer_1(inputs)
        # D_input = tf.concat([g_output, inputs], 2)
        output = self.layer_2(g_output)

        base = tf.reduce_sum((output - 1) ** 2)
        # if epoch<3000:
        # if epoch<150:
        # if epoch<150:
        #     lumbda0=0# 1e-3
        # else:

        lumbda0=1#1
        lumbda1 = 0.7  # 0.8      4
        lumbda2 = 0.2 # 0.2  # 0.3     90
        lumbda3 = 0
        # else:
        #     lumbda1=0.1
        #     lumbda2=0
        # else:
        #     lumbda1=4
        #     lumbda2=0

        # x=range(len(raw_data[0,:,:]))
        # plt.subplot(121)
        # plt.plot(x,raw_data[0,:,:],x,g_output[0,:,:])
        # plt.subplot(122)
        # plt.plot(x,raw_data[0,:,:],x,inputs[0,:,:])
        # plt.show()

        # simple loss
        ldist = tf.sqrt(tf.reduce_sum((g_output - raw_data)**2))
        lmax = tf.reduce_max(abs(g_output - raw_data))
        # cc=piece_CC(raw_data,g_output)
        # print('cc=',cc)
        N = 512
        batch_size=128
        div_size=512
        # baseline_loss=np.zeros(batch_size)
        # for i in range(batch_size):
        #     x=g_output.numpy()[i,:,:]-raw_data[i,:,:]
        #     baseline_loss[i] = abs(np.fft.fftshift(np.fft.fft(x)))[0,0]
        # baseline_loss=np.sum(baseline_loss)
        #
        # baseline_loss = tf.cast(baseline_loss, dtype=tf.float32)
        print("Gloss base=",base," ldist=",ldist," lmax=",lmax)
        # print("baseline loss=",baseline_loss)
        loss = lumbda0*base+lumbda1 * ldist + lumbda2 * lmax

        # if lmax>1:
        # if epoch==4:
        #     print("show the train samples")
        #     for j in range(256):
        #         ground_truth_plot = raw_data[j, :]
        #         denoised_signal_plot = g_output[j, :]
        #         noisy_input_plot = inputs[j, :]
        #         print(('loss error denoise plot_' + str(j)))
        #         plt.subplot(221)
        #         plt.plot(ground_truth_plot)
        #         plt.ylim([-0.1, 1.1])
        #         plt.title('ground truth')
        #         plt.subplot(222)
        #         plt.plot(noisy_input_plot)
        #         plt.ylim([-0.1, 1.1])
        #         plt.title('noisy signal')
        #         plt.subplot(223)
        #         plt.plot(denoised_signal_plot)
        #         plt.ylim([-0.1, 1.1])
        #         plt.title('denoise signal')
        #         plt.subplot(224)
        #         x = range(len(ground_truth_plot))
        #         plt.plot(x, ground_truth_plot, x, denoised_signal_plot)
        #         plt.ylim([-0.1, 1.1])
        #         plt.title('compare of ground truth and denoise signal')
        #         plt.tight_layout()
        #         plt.savefig(
        #             ('/root/denois_CGAN/train_fig/train_fig_piece_' + str(j) + '.png'))
        #         plt.show()
        #         plt.close()

        # cus loss
        # kernel_size=48
        # sigma=kernel_size/6
        # amp=0.04
        # threshold=0
        # p_wave_kernel=kernel_gen(kernel_size,sigma,amp)
        # d=raw_data-g_output
        # p_wave_possi=abs(batch_conv(d,kernel=p_wave_kernel))
        #
        # # if t==1:
        # #     plt.plot(p_wave_possi[0])
        # #     plt.show()
        #
        # adjust_factor=1
        # ldist = tf.sqrt(tf.reduce_sum((1+adjust_factor*p_wave_possi)*abs(g_output - raw_data)))
        # lmax = tf.reduce_max((1+adjust_factor*p_wave_possi)*abs(g_output - raw_data))
        # print("Gloss base=",base," ldist=",ldist," lmax=",lmax)
        # loss = base+lumbda1 * ldist + lumbda2 * lmax
        return loss

    def D_loss(self, gen_input, real_input, epoch):
        gen_output = self.layer_2(gen_input)
        real_output=self.layer_2(real_input)

        rand_label_collid=0.05*np.random.normal(1)

        base = tf.reduce_sum((gen_output-0.1+rand_label_collid) ** 2
                              +(real_output-0.9+rand_label_collid)**2)/2
        lamuda=1
        loss=base*lamuda
        print("\nDloss loss=",loss)
        # D_loss=np.asarray([loss])
        # pd.DataFrame(D_loss).to_csv(
        #     ('/root/autodl-tmp/project/train_process/D_loss_history'+str(epoch)+'.csv'))
        return loss

    def G_get_grad(self, input, raw_data,epoch):
        with tf.GradientTape() as tape:
            tape.watch(self.layer_1.variables)
            L = self.G_loss(input,raw_data,epoch)
            g = tape.gradient(L, self.layer_1.variables)
        return g

    def D_get_grad(self, gen_input, real_input,epoch):
        with tf.GradientTape() as tape:
            tape.watch(self.layer_2.variables)
            L = self.D_loss(gen_input, real_input,epoch)
            g = tape.gradient(L, self.layer_2.variables)
        return g

    def training(self, input, testset, epochs,lr_G,lr_D,batch_size=256):
        # normed_trainset=input
        # every sample normalization
        # normed_trainset,leftout=piece_norm(input)

        # normed_testset,leftout=piece_norm(testset)

        normed_trainset=input # trainset
        normed_testset=testset

        # batch count
        train_num = normed_trainset.shape[0]
        datadim = normed_trainset.shape[1]
        channel = normed_trainset.shape[2]
        batch_count = int(train_num / batch_size)

        # data batchlization
        batched_trainset = data_batchlization(normed_trainset,
                                              batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)
        batched_testset=data_batchlization(normed_testset,
                                           batch_size)

        # data batchlization
        # batched_trainset = data_batchlization_gen_noise_try(normed_trainset,
        #                                       batch_size,SNR=0)  # batched_data.shape=(batch_index,batchsize,datadim,channel)


        # dataset reshape
        noise_data = batched_trainset[:, :, :, 1]
        noise_data = np.reshape(noise_data, (batch_count, batch_size, datadim, channel-1))
        clean_data = batched_trainset[:, :, :, 0]
        clean_data = np.reshape(clean_data, (batch_count, batch_size, datadim, 1))

        # CGAN definition
        generator = self.layer_1
        discriminator = self.layer_2
        gan=self
        self.compile()
        # gan = create_gan(discriminator, generator)

        # CGAN training
        for e in range(1, epochs + 1):
            print("Epoch %d" % e)
            # if e==150:# 20:
            #     lr_G=0.1*lr_G
            #     lr_D=0.1*lr_D
            # if e==70:
            #     lr_G=0.1*lr_G
            #     lr_D=0.5*lr_D
            if e==12:
                lr_G=0.1*lr_G
                lr_D=0.5*lr_D
            for _ in tqdm(range(int(batch_count))):
                noisedata_this_batch = noise_data[_, :, :, :]
                # noisedata_this_batch=np.reshape(noisedata_this_batch,(batch_size,datadim,channel))
                cleandata_this_batch = clean_data[_, :, :, :]
                # cleandata_this_batch=np.reshape(cleandata_this_batch,(batch_size,datadim,channel))

                # give noisy ECG as generator input
                noisy_ECG = noisedata_this_batch  # noisy_ECG.shape=(batch_size,time,1)

                # generator denoise
                denoise_ECG = generator.predict(noisy_ECG)  # denoise_ECG.shape=(batch_size,time,1)

                # fatch out the clean match ECG
                clean_ECG = cleandata_this_batch  # clean_ECG.shape=(batch_size,time,1)

                # concatenate the clean ECG and noisy ECG and denoise ECG
                # X_real_noisy = np.concatenate([clean_ECG, noisy_ECG], axis=2)  # X_real_noisy.shape=(batch_size,time,2)
                # X_gen_noisy = np.concatenate([denoise_ECG, noisy_ECG], axis=2)
                X_real_noisy = clean_ECG  # X_real_noisy.shape=(batch_size,time,2)
                X_gen_noisy = denoise_ECG

                training_strategy=1 # training_stratege is to set
                # 1:mix training(mix pos & neg sampleto train D) or
                # 0:class training(random select a class, pos or neg to train D each batch)
                if training_strategy:
                    # randomly choose a combination as the input of D
                    mix = np.concatenate([X_real_noisy, X_gen_noisy], axis=0)
                    mix_label = np.ones(2 * batch_size) # -0.1*np.random.randint(0,1,1)
                    mix_label[batch_size:] = 0  # +0.1*np.random.randint(0,1,1)

                    rand_index=np.arange(2*batch_size)
                    np.random.shuffle(rand_index)
                    mix_label=mix_label[rand_index]
                    mix=mix[rand_index]
                    X=mix
                    y_dis=mix_label
                else:
                    rand=np.random.randint(0,1,1)
                    if rand:
                        X=X_gen_noisy
                        y_dis=np.zeros(batch_size)
                    else:
                        X=X_real_noisy
                        y_dis=np.ones(batch_size)

                # pretraining for discriminator to tell the true or false
                self.layer_1.trainable=False
                self.layer_2.trainable=True

                if True:# e<5: #%1==0: # and e>=3:
                    grad_D = self.D_get_grad(X_gen_noisy,X_real_noisy,epoch=_)
                    tf.keras.optimizers.RMSprop(lr_D).apply_gradients(zip(grad_D, self.layer_2.variables))

                # D valid
                y_pred = np.squeeze(self.layer_2.predict(X,batch_size=128))
                y_pred[y_pred > 0.5] = 1  # 0.9
                y_pred[y_pred < 0.5] = 0  # 0.1
                if training_strategy:
                    y_dis[y_dis>0.5]=1
                    y_dis[y_dis<0.5]=0
                else:
                    y_dis=y_dis*np.ones(batch_size)
                acc = np.sum(y_dis == y_pred)/(2*batch_size)
                # acc = (y_dis == y_pred).astype(float).mean()
                print("acc=", acc)

                if True:# e<=5 or e%3==0: # or (e==1 and _<=15):
                    # input the denoise combination into fixed D and get the output of D
                    noisy_gan_input = noisy_ECG
                    y_gen_train = np.ones(batch_size)

                    # fix the discriminator when training GAN
                    self.layer_2.trainable = False
                    self.layer_1.trainable= True

                    # GAN train
                    grad_G = self.G_get_grad(noisy_gan_input,raw_data=clean_ECG,epoch=_)
                    tf.keras.optimizers.RMSprop(learning_rate=lr_G).apply_gradients(zip(grad_G, self.layer_1.variables))
            snr=plot_epoch_denoised_signals(batched_trainset, batched_testset, e, generator, batch_count)

            # save the best model
            '''
            if e >= 2:
                prio_metric = pd.read_csv(
                    ('/home/zhanghaowei/project/denoiseCGAN/train_process/train_process_' + str(e - 1) + '.csv'))
                prio_metric = prio_metric.to_numpy()
                prio_snr = prio_metric[0, 1]
                if prio_snr < snr:
                    generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
                    discriminator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_D')
                    # gan.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_CGAN')
            else:
                generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
                discriminator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_D')
                # gan.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_CGAN')
            '''
        # generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G.h5')
        #     self.save_weights('/root/autodl-tmp/project/denoise_core/model_weight_1')
        #     generator.save('/root/autodl-tmp/project/denoise_core/denoise_G')
        # generator.save('/home/uu201912628/project/denoise_core/denoise_G')
        generator.save('/home/zhanghaowei/DDPM/saved_model/gan')

    def training_ddpm_adapt(self, train_clean, train_noisy, test_clean, test_noisy,epochs,lr_G,lr_D,batch_size=256):
        # normed_trainset=input
        # every sample normalization
        # normed_trainset,leftout=piece_norm(input)

        # normed_testset,leftout=piece_norm(testset)
        batch_count = train_clean.shape[0]

        # CGAN definition
        generator = self.layer_1
        discriminator = self.layer_2
        gan=self
        self.compile()
        # gan = create_gan(discriminator, generator)

        # CGAN training
        for e in range(1, epochs + 1):
            print("Epoch %d" % e)
            # if e==150:# 20:
            #     lr_G=0.1*lr_G
            #     lr_D=0.1*lr_D
            # if e==70:
            #     lr_G=0.1*lr_G
            #     lr_D=0.5*lr_D
            if e==12:
                lr_G=0.1*lr_G
                lr_D=0.5*lr_D
            for _ in tqdm(range(int(batch_count))):
                print('batch num=', _)
                noisedata_this_batch = train_noisy[_, :, :, :]
                # noisedata_this_batch=np.reshape(noisedata_this_batch,(batch_size,datadim,channel))
                cleandata_this_batch = train_clean[_, :, :, :]
                # cleandata_this_batch=np.reshape(cleandata_this_batch,(batch_size,datadim,channel))

                # give noisy ECG as generator input
                noisy_ECG = noisedata_this_batch  # noisy_ECG.shape=(batch_size,time,1)

                # generator denoise
                denoise_ECG = generator.predict(noisy_ECG)  # denoise_ECG.shape=(batch_size,time,1)

                # fatch out the clean match ECG
                clean_ECG = cleandata_this_batch  # clean_ECG.shape=(batch_size,time,1)

                # concatenate the clean ECG and noisy ECG and denoise ECG
                # X_real_noisy = np.concatenate([clean_ECG, noisy_ECG], axis=2)  # X_real_noisy.shape=(batch_size,time,2)
                # X_gen_noisy = np.concatenate([denoise_ECG, noisy_ECG], axis=2)
                X_real_noisy = clean_ECG  # X_real_noisy.shape=(batch_size,time,2)
                X_gen_noisy = denoise_ECG

                training_strategy=1 # training_stratege is to set
                # 1:mix training(mix pos & neg sampleto train D) or
                # 0:class training(random select a class, pos or neg to train D each batch)
                if training_strategy:
                    # randomly choose a combination as the input of D
                    mix = np.concatenate([X_real_noisy, X_gen_noisy], axis=0)
                    mix_label = np.ones(2 * batch_size) # -0.1*np.random.randint(0,1,1)
                    mix_label[batch_size:] = 0  # +0.1*np.random.randint(0,1,1)

                    rand_index=np.arange(2*batch_size)
                    np.random.shuffle(rand_index)
                    mix_label=mix_label[rand_index]
                    mix=mix[rand_index]
                    X=mix
                    y_dis=mix_label
                else:
                    rand=np.random.randint(0,1,1)
                    if rand:
                        X=X_gen_noisy
                        y_dis=np.zeros(batch_size)
                    else:
                        X=X_real_noisy
                        y_dis=np.ones(batch_size)

                # pretraining for discriminator to tell the true or false
                self.layer_1.trainable=False
                self.layer_2.trainable=True

                if True:# e<5: #%1==0: # and e>=3:
                    grad_D = self.D_get_grad(X_gen_noisy,X_real_noisy,epoch=_)
                    tf.keras.optimizers.RMSprop(lr_D).apply_gradients(zip(grad_D, self.layer_2.variables))

                # D valid
                y_pred = np.squeeze(self.layer_2.predict(X,batch_size=128))
                y_pred[y_pred > 0.5] = 1  # 0.9
                y_pred[y_pred < 0.5] = 0  # 0.1
                if training_strategy:
                    y_dis[y_dis>0.5]=1
                    y_dis[y_dis<0.5]=0
                else:
                    y_dis=y_dis*np.ones(batch_size)
                acc = np.sum(y_dis == y_pred)/(2*batch_size)
                # acc = (y_dis == y_pred).astype(float).mean()
                print("acc=", acc)

                if True:# e<=5 or e%3==0: # or (e==1 and _<=15):
                    # input the denoise combination into fixed D and get the output of D
                    noisy_gan_input = noisy_ECG
                    y_gen_train = np.ones(batch_size)

                    # fix the discriminator when training GAN
                    self.layer_2.trainable = False
                    self.layer_1.trainable= True

                    # GAN train
                    grad_G = self.G_get_grad(noisy_gan_input,raw_data=clean_ECG,epoch=_)
                    tf.keras.optimizers.RMSprop(learning_rate=lr_G).apply_gradients(zip(grad_G, self.layer_1.variables))
                # snr=plot_epoch_denoised_signals(train_noisy, train_clean,test_noisy,test_clean, e, generator, batch_count)

            # save the best model
            '''
            if e >= 2:
                prio_metric = pd.read_csv(
                    ('/home/zhanghaowei/project/denoiseCGAN/train_process/train_process_' + str(e - 1) + '.csv'))
                prio_metric = prio_metric.to_numpy()
                prio_snr = prio_metric[0, 1]
                if prio_snr < snr:
                    generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
                    discriminator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_D')
                    # gan.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_CGAN')
            else:
                generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
                discriminator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_D')
                # gan.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_CGAN')
            '''
        # generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G.h5')
        #     self.save_weights('/root/autodl-tmp/project/denoise_core/model_weight_1')
        #     generator.save('/root/autodl-tmp/project/denoise_core/denoise_G')
        # generator.save('/home/uu201912628/project/denoise_core/denoise_G')
        generator.save('/home/zhanghaowei/DDPM/saved_model/gan')

    def trick_training(self, input, epochs,lr_G,lr_D,batch_size=256):
        # normed_trainset=input
        # every sample normalization
        # normed_trainset,leftout=piece_norm(input)
        # normed_trainset=trainset

        # batch count
        trainset=input
        train_num = trainset.shape[1]
        datadim = trainset.shape[2]
        channel = trainset.shape[3]
        batch_count = trainset.shape[0]# int(train_num / batch_size)

        # data batchlization
        # batched_trainset = data_batchlization(normed_trainset,
        #                                       batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

        # data batchlization
        # batched_trainset = data_batchlization_gen_noise_try(normed_trainset,
        #                                       batch_size,SNR=0)  # batched_data.shape=(batch_index,batchsize,datadim,channel)


        # dataset reshape
        noise_data = trainset[:, :, :, 1]
        noise_data = np.reshape(noise_data, (batch_count, batch_size, datadim, 1))
        clean_data = trainset[:, :, :, 0]
        clean_data = np.reshape(clean_data, (batch_count, batch_size, datadim, 1))

        # CGAN definition
        generator = self.layer_1
        discriminator = self.layer_2
        gan=self
        self.compile()
        # gan = create_gan(discriminator, generator)

        # CGAN training
        for e in range(1, epochs + 1):
            print("Epoch %d" % e)
            # if e==15:# 20:
            #     lr_G=0.1*lr_G
            #     lr_D=0.1*lr_D
            # if e==70:
            #     lr_G=0.1*lr_G
            #     lr_D=0.5*lr_D
            # if e==15:
            #     lr_G=0.1*lr_G
            #     lr_D=0.5*lr_D
            for _ in tqdm(range(int(batch_count))):
                noisedata_this_batch = noise_data[_, :, :, :]
                # noisedata_this_batch=np.reshape(noisedata_this_batch,(batch_size,datadim,channel))
                cleandata_this_batch = clean_data[_, :, :, :]
                # cleandata_this_batch=np.reshape(cleandata_this_batch,(batch_size,datadim,channel))

                # give noisy ECG as generator input
                noisy_ECG = noisedata_this_batch  # noisy_ECG.shape=(batch_size,time,1)

                # generator denoise
                denoise_ECG = generator.predict(noisy_ECG)  # denoise_ECG.shape=(batch_size,time,1)

                # fatch out the clean match ECG
                clean_ECG = cleandata_this_batch  # clean_ECG.shape=(batch_size,time,1)

                # concatenate the clean ECG and noisy ECG and denoise ECG
                # X_real_noisy = np.concatenate([clean_ECG, noisy_ECG], axis=2)  # X_real_noisy.shape=(batch_size,time,2)
                # X_gen_noisy = np.concatenate([denoise_ECG, noisy_ECG], axis=2)
                X_real_noisy = clean_ECG  # X_real_noisy.shape=(batch_size,time,2)
                X_gen_noisy = denoise_ECG

                training_strategy=1 # training_stratege is to set
                # 1:mix training(mix pos & neg sampleto train D) or
                # 0:class training(random select a class, pos or neg to train D each batch)
                if training_strategy:
                    # randomly choose a combination as the input of D
                    mix = np.concatenate([X_real_noisy, X_gen_noisy], axis=0)
                    mix_label = np.ones(2 * batch_size) # -0.1*np.random.randint(0,1,1)
                    mix_label[batch_size:] = 0  # +0.1*np.random.randint(0,1,1)

                    rand_index=np.arange(2*batch_size)
                    # np.random.shuffle(rand_index)
                    mix_label=mix_label[rand_index]
                    mix=mix[rand_index]
                    X=mix
                    y_dis=mix_label
                else:
                    rand=np.random.randint(0,1,1)
                    if rand:
                        X=X_gen_noisy
                        y_dis=np.zeros(batch_size)
                    else:
                        X=X_real_noisy
                        y_dis=np.ones(batch_size)

                # pretraining for discriminator to tell the true or false
                self.layer_1.trainable=False
                self.layer_2.trainable=True

                if e%1==0: # and e>=3:
                    grad_D = self.D_get_grad(X_gen_noisy,X_real_noisy,epoch=_)
                    tf.keras.optimizers.RMSprop(lr_D).apply_gradients(zip(grad_D, self.layer_2.variables))

                # D valid
                y_pred = np.squeeze(self.layer_2.predict(X,batch_size=128))
                y_pred[y_pred > 0.5] = 1  # 0.9
                y_pred[y_pred < 0.5] = 0  # 0.1
                if training_strategy:
                    y_dis[y_dis>0.5]=1
                    y_dis[y_dis<0.5]=0
                else:
                    y_dis=y_dis*np.ones(batch_size)
                acc = (y_dis == y_pred).astype(float).mean()
                print("acc=", acc)

                if e%1==0: # or (e==1 and _<=15):
                    # input the denoise combination into fixed D and get the output of D
                    noisy_gan_input = noisy_ECG
                    y_gen_train = np.ones(batch_size)

                    # fix the discriminator when training GAN
                    self.layer_2.trainable = False
                    self.layer_1.trainable=True

                    # GAN train
                    grad_G = self.G_get_grad(noisy_gan_input,raw_data=clean_ECG,epoch=_)
                    tf.keras.optimizers.RMSprop(learning_rate=lr_G).apply_gradients(zip(grad_G, self.layer_1.variables))
            snr=plot_epoch_denoised_signals(trainset, e, generator, batch_count)

            # save the best model
            '''
            if e >= 2:
                prio_metric = pd.read_csv(
                    ('/home/zhanghaowei/project/denoiseCGAN/train_process/train_process_' + str(e - 1) + '.csv'))
                prio_metric = prio_metric.to_numpy()
                prio_snr = prio_metric[0, 1]
                if prio_snr < snr:
                    generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
                    discriminator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_D')
                    # gan.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_CGAN')
            else:
                generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
                discriminator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_D')
                # gan.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_CGAN')
            '''
        # generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G.h5')
        generator.save('/root/autodl-tmp/project/denoise_core/denoise_G')
        # generator.save('/root/denois_CGAN/denoise_core/denoise_G')

def model_reload_training(trainset,testset,lr_G=1e-4,lr_D=1e-4,batch_size=128):
    # model reload
    denoise_GAN = CGAN()# tf.keras.models.load_model('/root/autodl-tmp/project/denoise_core/denoise_G')
    denoise_GAN.load_weights('/root/denois_CGAN/denoise_core/model_weight_2')

    normed_trainset = trainset  # trainset
    normed_testset = testset

    # batch count
    train_num = normed_trainset.shape[0]
    datadim = normed_trainset.shape[1]
    channel = normed_trainset.shape[2]
    batch_count = int(train_num / batch_size)

    # data batchlization
    batched_trainset = data_batchlization(normed_trainset,
                                          batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)
    batched_testset = data_batchlization(normed_testset,
                                         batch_size)

    # data batchlization
    # batched_trainset = data_batchlization_gen_noise_try(normed_trainset,
    #                                       batch_size,SNR=0)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # dataset reshape
    noise_data = batched_trainset[:, :, :, 1]
    noise_data = np.reshape(noise_data, (batch_count, batch_size, datadim, channel - 1))
    clean_data = batched_trainset[:, :, :, 0]
    clean_data = np.reshape(clean_data, (batch_count, batch_size, datadim, 1))

    # CGAN definition
    generator = denoise_GAN.layer_1
    discriminator = denoise_GAN.layer_2
    # discriminator.compile()
    # gan = create_gan(discriminator, generator)
    epochs=10000
    # CGAN training
    for e in range(1, epochs + 1):
        print("Epoch %d" % e)
        # if e==150:# 20:
        #     lr_G=0.1*lr_G
        #     lr_D=0.1*lr_D
        # if e==70:
        #     lr_G=0.1*lr_G
        #     lr_D=0.5*lr_D
        # if e==15:
        #     lr_G=0.1*lr_G
        #     lr_D=0.5*lr_D
        for _ in tqdm(range(int(batch_count))):
            noisedata_this_batch = noise_data[_, :, :, :]
            # noisedata_this_batch=np.reshape(noisedata_this_batch,(batch_size,datadim,channel))
            cleandata_this_batch = clean_data[_, :, :, :]
            # cleandata_this_batch=np.reshape(cleandata_this_batch,(batch_size,datadim,channel))

            # give noisy ECG as generator input
            noisy_ECG = noisedata_this_batch  # noisy_ECG.shape=(batch_size,time,1)

            # generator denoise
            denoise_ECG = generator.predict(noisy_ECG)  # denoise_ECG.shape=(batch_size,time,1)

            # fatch out the clean match ECG
            clean_ECG = cleandata_this_batch  # clean_ECG.shape=(batch_size,time,1)

            # concatenate the clean ECG and noisy ECG and denoise ECG
            # X_real_noisy = np.concatenate([clean_ECG, noisy_ECG], axis=2)  # X_real_noisy.shape=(batch_size,time,2)
            # X_gen_noisy = np.concatenate([denoise_ECG, noisy_ECG], axis=2)
            X_real_noisy = clean_ECG  # X_real_noisy.shape=(batch_size,time,2)
            X_gen_noisy = denoise_ECG

            training_strategy = 0  # training_stratege is to set
            # 1:mix training(mix pos & neg sampleto train D) or
            # 0:class training(random select a class, pos or neg to train D each batch)
            if training_strategy:
                # randomly choose a combination as the input of D
                mix = np.concatenate([X_real_noisy, X_gen_noisy], axis=0)
                mix_label = np.ones(2 * batch_size)  # -0.1*np.random.randint(0,1,1)
                mix_label[batch_size:] = 0  # +0.1*np.random.randint(0,1,1)

                rand_index = np.arange(2 * batch_size)
                # np.random.shuffle(rand_index)
                mix_label = mix_label[rand_index]
                mix = mix[rand_index]
                X = mix
                y_dis = mix_label
            else:
                rand = np.random.randint(0, 1, 1)
                if rand:
                    X = X_gen_noisy
                    y_dis = np.zeros(batch_size)
                else:
                    X = X_real_noisy
                    y_dis = np.ones(batch_size)

            # pretraining for discriminator to tell the true or false
            generator.trainable = False
            discriminator.trainable = True

            if e % 1 == 0:  # and e>=3:
                grad_D = denoise_GAN.D_get_grad(X_gen_noisy, X_real_noisy, epoch=_)
                tf.keras.optimizers.RMSprop(lr_D).apply_gradients(zip(grad_D, discriminator.variables))

            # D valid
            y_pred = np.squeeze(discriminator.predict(X, batch_size=128))
            y_pred[y_pred > 0.5] = 1  # 0.9
            y_pred[y_pred < 0.5] = 0  # 0.1
            if training_strategy:
                y_dis[y_dis > 0.5] = 1
                y_dis[y_dis < 0.5] = 0
            else:
                y_dis = y_dis * np.ones(batch_size)
            acc = (y_dis == y_pred).astype(float).mean()
            print("acc=", acc)

            if e % 1 == 0:  # or (e==1 and _<=15):
                # input the denoise combination into fixed D and get the output of D
                noisy_gan_input = noisy_ECG
                y_gen_train = np.ones(batch_size)

                # fix the discriminator when training GAN
                discriminator.trainable = False
                generator.trainable = True

                # GAN train
                grad_G = denoise_GAN.G_get_grad(noisy_gan_input, raw_data=clean_ECG, epoch=_)
                tf.keras.optimizers.RMSprop(learning_rate=lr_G).apply_gradients(zip(grad_G, generator.variables))
        snr = plot_epoch_denoised_signals(batched_trainset, batched_testset, e, generator, batch_count)

        # generator.save('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G.h5')
        # denoise_GAN.save_weights('/root/autodl-tmp/project/denoise_core/model_weight_2')
        # generator.save('/root/autodl-tmp/project/denoise_core/denoise_G')
    # generator.save('/root/denois_CGAN/denoise_core/denoise_G')

def denoise_CGAN_denoise(noisy_data, recover_bar,batch_size=256):
    batch_size=noisy_data.shape[1]
    datadim=noisy_data.shape[2]
    channel=noisy_data.shape[3]

    # batch count
    batch_num=noisy_data.shape[0] # int(sample_num/batch_size)

    # trained model load
    # denoise_model=tf.keras.models.load_model('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
    # denoise_model = tf.keras.models.load_model('/root/autodl-tmp/project/denoise_core/denoise_G')
    # denoise_model = tf.keras.models.load_model('/home/uu201912628/project/denoise_core/denoise_G')
    # denoise_model = tf.keras.models.load_model('/home/zhanghaowei/DDPM/saved_model/cae_gan')
    denoise_model = tf.keras.models.load_model('/home/zhanghaowei/DDPM/saved_model/encoder_model')

    # data batchilization
    # batched_testset = data_batchlization(noisy_data, batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # model predict for noise data
    # noise_data=batched_testset[:,:,:,1]
    # noise_data=np.reshape(noise_data,(batch_num,batch_size,datadim,1))

    # initialization
    denoise_data=np.zeros((batch_num,batch_size,datadim,12))

    # denoising
    for i in range(batch_num):
        # for k in range(recurrent_valid):
        denoise_data[i,:,:,:]=denoise_model.predict(noisy_data[i,:,:,:])
        # for j in range(batch_size):
        #     denoise_data[i,j,:,:]=denoise_data[i,j,:,:]+recover_bar[i*batch_size+j,0]

    return denoise_data

def denoise_DNN_CGAN_denoise(noisy_data, recover_bar,batch_size=256):
    batch_size=noisy_data.shape[1]
    datadim=noisy_data.shape[2]
    channel=noisy_data.shape[3]

    # batch count
    batch_num=noisy_data.shape[0] # int(sample_num/batch_size)

    # trained model load
    # denoise_model=tf.keras.models.load_model('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
    # denoise_model = tf.keras.models.load_model('/root/autodl-tmp/project/denoise_core/denoise_G')
    # denoise_model = tf.keras.models.load_model('/home/uu201912628/project/denoise_core/denoise_G')
    denoise_model = tf.keras.models.load_model('/root/denois_CGAN/denoise_core/denoise_G')

    # data batchilization
    # batched_testset = data_batchlization(noisy_data, batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # model predict for noise data
    # noise_data=batched_testset[:,:,:,1]
    # noise_data=np.reshape(noise_data,(batch_num,batch_size,datadim,1))

    # initialization
    noisy_data=noisy_data[:,:,:,0]
    denoise_data=np.zeros((batch_num,batch_size,datadim))

    # denoising
    for i in range(batch_num):
        # for k in range(recurrent_valid):
        denoise_data[i,:,:]=denoise_model.predict(noisy_data[i,:,:])
        # for j in range(batch_size):
        #     denoise_data[i,j,:,:]=denoise_data[i,j,:,:]+recover_bar[i*batch_size+j,0]

    return denoise_data


def denoise_CGAN_valid(testset, recover_bar,batch_size=256):
    sample_num=testset.shape[0]
    datadim=testset.shape[1]
    channel=testset.shape[2]

    # batch count
    batch_num=int(sample_num/batch_size)

    # trained model load
    # denoise_model=tf.keras.models.load_model('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
    # denoise_model = tf.keras.models.load_model('/root/autodl-tmp/project/denoise_core/denoise_G')
    denoise_model = tf.keras.models.load_model('/home/uu201912628/project/denoise_core/denoise_G')

    # sample normalization
    # normed_testset,normed_recov_bar=piece_norm(testset)
    # normed_testset=testset

    # data batchilization
    batched_testset = data_batchlization(testset, batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # batched_testset = data_batchlization_gen_noise_try(normed_testset,
    #                                                     batch_size,
    #                                                     SNR=0)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # model predict for noise data
    noise_data=batched_testset[:,:,:,1]

    # label
    ground_truth=batched_testset[:,:,:,0]

    noise_data=np.reshape(noise_data,(batch_num,batch_size,datadim,1))

    # initialization
    denoise_data=np.zeros((batch_num,batch_size,datadim,1))

    # denoising
    recurrent_valid=1
    for i in range(batch_num):
        # for k in range(recurrent_valid):
        denoise_data[i,:,:,:]=denoise_model.predict(noise_data[i,:,:,:])
        for j in range(batch_size):
            denoise_data[i,j,:,:]=denoise_data[i,j,:,:]+recover_bar[i*batch_size+j,0]
            ground_truth[i,j,:]=ground_truth[i,j,:]+recover_bar[i*batch_size+j]
        # denoise_data[i,:,:,:]=noise_data[i,:,:,:]



    # data=np.load('/root/autodl-tmp/project/code/trick_train_denoi_dataset.npz')
    # trick_clean=data['arr_0']
    # trick_clean=trick_clean[:,:,:,0]
    # trick_clean=np.reshape(trick_clean,(batch_num,batch_size,datadim,1))
    # trick_train_dataset=np.concatenate([trick_clean,noise_data],axis=3)
    # np.savez('/root/autodl-tmp/project/code/trick_train_dataset.npz',trick_train_dataset)


    # trick_dataset=np.concatenate([denoise_data,noise_data],axis=3)
    # np.savez('/root/autodl-tmp/project/code/trick_test_denoi_dataset.npz',denoise_data)




    # full plot
    # for i in range(batch_num):
    #     for j in range(batch_size):
    #         ground_truth_plot=ground_truth[i,j,:]
    #         denoised_signal_plot=denoise_data[i,j,:]
    #         noisy_input_plot=noise_data[i,j,:]
    #         print(('valid denoise plot_'+str(i)+'_'+str(j)))
    #         plt.subplot(221)
    #         plt.plot(ground_truth_plot)
    #         plt.ylim([-0.1, 1.1])
    #         plt.title('ground truth')
    #         plt.subplot(222)
    #         plt.plot(noisy_input_plot)
    #         # plt.ylim([-0.1, 1.1])
    #         plt.title('noisy signal')
    #         plt.subplot(223)
    #         plt.plot(denoised_signal_plot)
    #         plt.ylim([-0.1, 1.1])
    #         plt.title('denoise signal')
    #         plt.subplot(224)
    #         x = range(len(ground_truth_plot))
    #         plt.plot(x, ground_truth_plot, x, denoised_signal_plot)
    #         plt.ylim([-0.1, 1.1])
    #         plt.title('compare of ground truth and denoise signal')
    #         plt.tight_layout()
    #         plt.savefig(('/root/autodl-tmp/project/train_fig/valid_fig_batch'+str(i)+'_piece_'+str(j)+'.png'))
    #         # plt.show()
    #         plt.close()

    # denorm
    # denoise_data_debatch=np.reshape(denoise_data,(denoise_data.shape[0]*denoise_data.shape[1],datadim,1))
    # denoise_data_debatch=np.zeros((denoise_data.shape[0]*denoise_data.shape[1],datadim,1))
    # for b in range(denoise_data.shape[0]):
    #     denoise_data_debatch[b*batch_size:(b+1)*batch_size,:,:]=denoise_data[b,:,:]
    # denoise_data=piece_denorm(denoise_data_debatch,normed_recov_bar)
    # ground_truth=piece_denorm(ground_truth,normed_recov_bar)

    # reshape
    ground_truth=np.reshape(ground_truth,(ground_truth.shape[0]*ground_truth.shape[1]*ground_truth.shape[2],1))
    noise_data_plot=np.reshape(noise_data,(noise_data.shape[0]*noise_data.shape[1]*noise_data.shape[2],1))
    denoise_data=np.reshape(denoise_data,(denoise_data.shape[0]*denoise_data.shape[1]*denoise_data.shape[2],1))



    # reshape plot
    time_s=182500
    time_e=184500
    ground_truth_plot=ground_truth[time_s:time_e]
    noise_data_plot=noise_data_plot[time_s:time_e]
    denoise_data_plot=denoise_data[time_s:time_e]

    print(('full denoise plot'))
    plt.subplot(221)
    plt.plot(ground_truth_plot)
    plt.ylim([-1.1, 1.1])
    plt.title('ground truth')
    plt.subplot(222)
    plt.plot(noise_data_plot)
    plt.ylim([-1.1, 1.1])
    plt.title('noisy signal')
    plt.subplot(223)
    plt.plot(denoise_data_plot)
    plt.ylim([-1.1, 1.1])
    plt.title('denoise signal')
    plt.subplot(224)
    x = range(len(ground_truth_plot))
    plt.plot(x, ground_truth_plot, x, denoise_data_plot)
    plt.ylim([-1.1, 1.1])
    plt.title('compare of ground truth and denoise signal')
    plt.tight_layout()
    # plt.savefig(('/root/autodl-tmp/project/train_fig/valid_fig_batch' + str(i) + '_piece_' + str(j) + '.png'))
    plt.show()




    # performance evaluation
    snr=SNR(ground_truth,denoise_data)
    rmse=RMSE(ground_truth,denoise_data)
    # snr = SNR(ground_truth, noise_data_plot)
    # rmse = RMSE(ground_truth, noise_data_plot)
    result=np.asarray([snr,rmse])
    # pd.DataFrame(result).to_csv(('/home/zhanghaowei/project/denoiseCGAN/train_process/train_process_'+str(epoch)+'.csv'))
    # pd.DataFrame(result).to_csv(
    #     ('/root/autodl-tmp/project/train_process/valid_result.csv'))
    pd.DataFrame(result).to_csv(
        ('/home/uu201912628/project/train_process/valid_result.csv'))
    '''
    plt.subplot(221)
    plt.plot(ground_truth)
    plt.subplot(222)
    plt.plot(noise_data_plot)
    plt.subplot(223)
    plt.plot(denoise_data)
    plt.show()
    '''
    return snr,rmse


def denoise_CGAN_trick_valid(testset, batch_size=256):
    # sample_num=testset.shape[0]
    datadim=testset.shape[2]
    channel=testset.shape[3]

    # batch count
    batch_num=testset.shape[0]# int(sample_num/batch_size)

    # trained model load
    # denoise_model=tf.keras.models.load_model('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
    denoise_model = tf.keras.models.load_model('/root/autodl-tmp/project/denoise_core/denoise_G')
    # denoise_model = tf.keras.models.load_model('/root/denois_CGAN/denoise_core/denoise_G')

    # sample normalization
    # normed_testset,normed_recov_bar=piece_norm(testset)
    # normed_testset=testset

    # data batchilization
    # batched_testset = data_batchlization(normed_testset, batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # batched_testset = data_batchlization_gen_noise_try(normed_testset,
    #                                                     batch_size,
    #                                                     SNR=0)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # model predict for noise data
    noise_data=testset[:,:,:,1]

    # label
    ground_truth=testset[:,:,:,0]

    noise_data=np.reshape(noise_data,(batch_num,batch_size,datadim,1))

    # initialization
    denoise_data=np.zeros((batch_num,batch_size,datadim,1))

    # denoising
    recurrent_valid=1
    for i in range(batch_num):
        # for k in range(recurrent_valid):
        denoise_data[i,:,:,:]=denoise_model.predict(noise_data[i,:,:,:])
        # denoise_data[i,:,:,:]=noise_data[i,:,:,:]

    # performance evaluation
    snr=SNR(ground_truth,denoise_data)
    rmse=RMSE(ground_truth,denoise_data)


    # data=np.load('/root/autodl-tmp/project/code/trick_train_denoi_dataset.npz')
    # trick_clean=data['arr_0']
    # trick_clean=trick_clean[:,:,:,0]
    # trick_clean=np.reshape(trick_clean,(batch_num,batch_size,datadim,1))
    # trick_train_dataset=np.concatenate([trick_clean,noise_data],axis=3)
    # np.savez('/root/autodl-tmp/project/code/trick_train_dataset.npz',trick_train_dataset)


    # trick_dataset=np.concatenate([denoise_data,noise_data],axis=3)
    # np.savez('/root/autodl-tmp/project/code/trick_test_denoi_dataset.npz',denoise_data)




    # full plot
    # for i in range(batch_num):
    #     for j in range(batch_size):
    #         ground_truth_plot=ground_truth[i,j,:]
    #         denoised_signal_plot=denoise_data[i,j,:]
    #         noisy_input_plot=noise_data[i,j,:]
    #         print(('valid denoise plot_'+str(i)+'_'+str(j)))
    #         plt.subplot(221)
    #         plt.plot(ground_truth_plot)
    #         plt.ylim([-0.1, 1.1])
    #         plt.title('ground truth')
    #         plt.subplot(222)
    #         plt.plot(noisy_input_plot)
    #         plt.ylim([-0.1, 1.1])
    #         plt.title('noisy signal')
    #         plt.subplot(223)
    #         plt.plot(denoised_signal_plot)
    #         plt.ylim([-0.1, 1.1])
    #         plt.title('denoise signal')
    #         plt.subplot(224)
    #         x = range(len(ground_truth_plot))
    #         plt.plot(x, ground_truth_plot, x, denoised_signal_plot)
    #         plt.ylim([-0.1, 1.1])
    #         plt.title('compare of ground truth and denoise signal')
    #         plt.tight_layout()
    #         plt.savefig(('/root/autodl-tmp/project/train_fig/valid_fig_batch'+str(i)+'_piece_'+str(j)+'.png'))
    #         # plt.show()
    #         plt.close()

    # denorm
    # denoise_data_debatch=np.reshape(denoise_data,(denoise_data.shape[0]*denoise_data.shape[1],datadim,1))
    # denoise_data_debatch=np.zeros((denoise_data.shape[0]*denoise_data.shape[1],datadim,1))
    # for b in range(denoise_data.shape[0]):
    #     denoise_data_debatch[b*batch_size:(b+1)*batch_size,:,:]=denoise_data[b,:,:]
    # denoise_data=piece_denorm(denoise_data_debatch,normed_recov_bar)
    # ground_truth=piece_denorm(ground_truth,normed_recov_bar)

    # reshape
    ground_truth=np.reshape(ground_truth,(ground_truth.shape[0]*ground_truth.shape[1]*ground_truth.shape[2],1))
    noise_data_plot=np.reshape(noise_data,(noise_data.shape[0]*noise_data.shape[1]*noise_data.shape[2],1))
    denoise_data=np.reshape(denoise_data,(denoise_data.shape[0]*denoise_data.shape[1]*denoise_data.shape[2],1))



    # reshape plot
    time_s=182500
    time_e=183500
    ground_truth_plot=ground_truth[time_s:time_e]
    noise_data_plot=noise_data_plot[time_s:time_e]
    denoise_data_plot=denoise_data[time_s:time_e]

    # print(('full denoise plot'))
    # plt.subplot(221)
    # plt.plot(ground_truth_plot)
    # plt.ylim([-1.1, 1.1])
    # plt.title('ground truth')
    # plt.subplot(222)
    # plt.plot(noise_data_plot)
    # plt.ylim([-1.1, 1.1])
    # plt.title('noisy signal')
    # plt.subplot(223)
    # plt.plot(denoise_data_plot)
    # plt.ylim([-1.1, 1.1])
    # plt.title('denoise signal')
    # plt.subplot(224)
    # x = range(len(ground_truth_plot))
    # plt.plot(x, ground_truth_plot, x, denoise_data_plot)
    # plt.ylim([-1.1, 1.1])
    # plt.title('compare of ground truth and denoise signal')
    # plt.tight_layout()
    # # plt.savefig(('/root/autodl-tmp/project/train_fig/valid_fig_batch' + str(i) + '_piece_' + str(j) + '.png'))
    # plt.show()





    # snr = SNR(ground_truth, noise_data_plot)
    # rmse = RMSE(ground_truth, noise_data_plot)
    result=np.asarray([snr,rmse])
    # pd.DataFrame(result).to_csv(('/home/zhanghaowei/project/denoiseCGAN/train_process/train_process_'+str(epoch)+'.csv'))
    pd.DataFrame(result).to_csv(
        ('/root/autodl-tmp/project/train_process/valid_result.csv'))
    # pd.DataFrame(result).to_csv(
    #     ('/root/denois_CGAN/train_process/valid_result.csv'))
    '''
    plt.subplot(221)
    plt.plot(ground_truth)
    plt.subplot(222)
    plt.plot(noise_data_plot)
    plt.subplot(223)
    plt.plot(denoise_data)
    plt.show()
    '''
    return snr,rmse


