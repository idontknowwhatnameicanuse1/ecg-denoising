import numpy as np
import pandas as pd
import os
import metrics
import matplotlib.pyplot as plt
from ecgdetectors import Detectors


def nfold_dataset_gen(path_name,div_size,SNR,train_por,if_train,mix_por=np.array([1,0,0])):
    # data path
    DataDir_clean_ECG =(path_name+'/mit_ECG_csv/')
    DataDir_noise = (path_name+'/mit_noise_csv/')

    # noise reading
    Data_noise_BW = pd.read_csv((DataDir_noise+'bw.csv'))
    Data_noise_EM = pd.read_csv((DataDir_noise+'em.csv'))
    Data_noise_MA = pd.read_csv((DataDir_noise+'ma.csv'))
    Data_noise_BW = Data_noise_BW.to_numpy()
    Data_noise_EM = Data_noise_EM.to_numpy()
    Data_noise_MA = Data_noise_MA.to_numpy()

    BW = Data_noise_BW[:, 1]  # 1 is the easier
    BW = np.reshape(BW, (BW.shape[0], 1))

    EM = Data_noise_EM[:, 1]  # 1 is the easier
    EM = np.reshape(EM, (EM.shape[0], 1))

    MA = Data_noise_MA[:, 1]
    MA = np.reshape(MA, (MA.shape[0], 1))

    # clean ECG reading & match data generation
    match_sample=[]
    recover_bar_total=[]
    filenames=os.listdir(DataDir_clean_ECG)
    name_bank=['selected_ECG_219_MLII.csv','selected_ECG_111_MLII.csv',
               'selected_ECG_122_MLII.csv','selected_ECG_223_MLII.csv',
               'selected_ECG_116_MLII.csv','selected_ECG_213_MLII.csv',
               'selected_ECG_205_MLII.csv','selected_ECG_230_MLII.csv',
               'selected_ECG_103_MLII.csv','selected_ECG_105_MLII.csv'
               ]
    # 213 116
    nfile = len(name_bank)

    # random select certain num of files
    # np.random.shuffle(name_bank)

    snr_bank=np.array([0,1.25,5])
    mix_choice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                           [1, 1, 1]])
    # snr_bank=np.array([0])
    # mix_choice = np.array([[1, 0, 0]])

    # if_train=1
    # trainset gen
    if if_train:
        for i in range(int(nfile*train_por)):
            fulname = (DataDir_clean_ECG + name_bank[i])
            Data_clean_ECG = pd.read_csv(fulname)
            Data_clean_ECG = Data_clean_ECG.to_numpy()
            print('train_'+fulname)
            for j in range(snr_bank.shape[0]):
                for k in range(mix_choice.shape[0]):
                    snr=snr_bank[j]
                    train_mix_por=mix_choice[k,:]

                    # match data generation
                    match_sample_long = match_Datagen(Data_clean_ECG, BW, EM, MA,
                                                      snr, train_mix_por)

                    # divide the long data into pieces
                    sample, recover_bar = longdata_div(match_sample_long,
                                                       div_size)  # match_sample.shape=(sample_num,time,type)
                    if match_sample == []:
                        match_sample = sample
                        recover_bar_total = recover_bar
                    else:
                        match_sample = np.concatenate((match_sample, sample), axis=0)
                        recover_bar_total = np.concatenate((recover_bar_total, recover_bar), axis=0)

    # trainset gen
    if if_train:
        trainset,leftout,leftout=dataset_gen(match_sample, train_por=1,recover_bar=recover_bar_total)
    else:
        trainset=0

    # testset gen
    match_sample=[]
    for i in range(int(nfile*train_por),int(nfile)):
        # filenames gen & clean data read
        fulname=(DataDir_clean_ECG+name_bank[i])
        Data_clean_ECG=pd.read_csv(fulname)
        Data_clean_ECG=Data_clean_ECG.to_numpy()
        print(fulname)

        # match data generation
        match_sample_long=match_Datagen(Data_clean_ECG, BW, EM, MA,
                      SNR, mix_por)

        # divide the long data into pieces
        sample,recover_bar=longdata_div(match_sample_long,div_size)  # match_sample.shape=(sample_num,time,type)

        if match_sample==[]:
            match_sample=sample
            recover_bar_total = recover_bar
        else:
            match_sample=np.concatenate((match_sample,sample),axis=0)
            recover_bar_total = np.concatenate((recover_bar_total, recover_bar), axis=0)

    # testset gen
    testset,leftout,recover_bar_total=dataset_gen(match_sample, train_por=1,recover_bar=recover_bar_total)
    # testset=0
    return trainset,testset,recover_bar_total

def match_Datagen(Data_clean_ECG, Data_noise_BW, Data_noise_EM, Data_noise_MA, SNR, mix_por):
    # add the noise to clean ECG to create match data
    # noise gen
    # length=Data_clean_ECG.shape[0]
    noise=(Data_noise_BW*mix_por[0]+Data_noise_EM*mix_por[1]+Data_noise_MA*mix_por[2])/np.sum(mix_por)
    noise=noise[:Data_clean_ECG.shape[0],:]
    noise=noise# +0.2# 1.2

    # SNR adjust
    k=(np.sqrt(np.sum(Data_clean_ECG**2)/(np.sum(noise**2)*np.power(10,(SNR/10))))).astype('double')

    # match data gen
    Data_noisy = Data_clean_ECG+k*noise

    Data_clean_ECG = (Data_clean_ECG - np.min(Data_noisy)) / \
                 (np.max(Data_noisy) - np.min(Data_noisy))
    Data_noisy=(Data_noisy-np.min(Data_noisy))/\
               (np.max(Data_noisy)-np.min(Data_noisy))

    match_sample_long=np.concatenate((Data_clean_ECG,Data_noisy),axis=1)
    return match_sample_long

def longdata_div(long_data,div_size):
    # divide the long match data into pieces, long_data.shape=(data_length,data_type)
    # the 1st channel is data, 2nd channel is clean or noisy, 3rd channel is cos time encoding
    T=long_data.shape[0]
    piece_num=(np.floor(T/div_size)).astype('int')
    match_sample=np.zeros((piece_num,div_size,2))
    recover_bar=np.zeros((piece_num,1))
    for i in range(piece_num):
        # x0 and x1 is the clean ecg and the noisy one, respectively
        x0=long_data[i*div_size:(i+1)*div_size,0]   # the clean data
        x1 = long_data[i * div_size:(i + 1) * div_size, 1]    # the noisy data and the cos time encoding
        x0=np.reshape(x0,(div_size,1))
        x1=np.reshape(x1,(div_size,1))

        # the recover bar record the baseline shifted
        recover_bar[i,0]=np.real((1/div_size*np.fft.fft(x0,axis=0))[0])

        # shift the baseline to 0
        x0=x0-np.real((1/div_size*np.fft.fft(x0,axis=0))[0])
        match_sample[i, :, 0]=np.reshape(x0,(div_size))

        x1[:,0]=x1[:,0]-np.real((1/div_size*np.fft.fft(x1[:,0],axis=0))[0])   # baseline correction only for noisy data
        match_sample[i, :, 1]=np.reshape(x1,(div_size))
        # match_sample[i, :, 1] = x1
    return match_sample,recover_bar

def longdata_div_timebar_version(long_data,div_size):
    # divide the long match data into pieces, long_data.shape=(data_length,data_type)
    # the 1st channel is data, 2nd channel is clean or noisy, 3rd channel is cos time encoding
    T=long_data.shape[0]
    piece_num=(np.floor(T/div_size)).astype('int')
    match_sample=np.zeros((piece_num,div_size,3))
    recover_bar=np.zeros((piece_num,1))
    for i in range(piece_num):
        # x0 and x1 is the clean ecg and the noisy one, respectively
        x0=long_data[i*div_size:(i+1)*div_size,0]   # the clean data
        x1 = long_data[i * div_size:(i + 1) * div_size, 1]    # the noisy data and the cos time encoding
        time_bar=long_data[i * div_size:(i + 1) * div_size, 2]

        x0=np.reshape(x0,(div_size,1))
        x1=np.reshape(x1,(div_size,1))
        time_bar=np.reshape(time_bar,(div_size,1))

        # the recover bar record the baseline shifted
        recover_bar[i,0]=np.real((1/div_size*np.fft.fft(x0,axis=0))[0])

        # shift the baseline to 0
        x0=x0-np.real((1/div_size*np.fft.fft(x0,axis=0))[0])
        match_sample[i, :, 0]=np.reshape(x0,(div_size))

        x1[:,0]=x1[:,0]-np.real((1/div_size*np.fft.fft(x1[:,0],axis=0))[0])   # baseline correction only for noisy data
        match_sample[i, :, 1]=np.reshape(x1,(div_size))
        match_sample[i, :, 2] = np.reshape(time_bar,(div_size))
    return match_sample,recover_bar

def dataset_gen(match_sample,train_por,recover_bar):
    # generate trainset and testset from total dataset
    num=match_sample.shape[0]
    train_num=np.floor(num*train_por).astype('int')

    # randomly choose sample from total data
    index_rand_array = np.arange(match_sample.shape[0])
    # np.random.shuffle(index_rand_array)
    trainset = match_sample[index_rand_array[:train_num]]
    testset=match_sample[index_rand_array[train_num:]]
    recover_bar=recover_bar[index_rand_array]
    return trainset, testset, recover_bar

def data_batchlization(data,batchsize):
    # divide the dataset into batches, left out the final samples which not a full batch
    sample_num=data.shape[0]
    datadim=data.shape[1]
    channel=data.shape[-1]
    batch_num=np.floor(sample_num/batchsize).astype('int')
    batched_data=np.zeros((batch_num,batchsize,datadim,channel))
    for i in range(batch_num):
        batched_data[i,:,:,:]=data[i*batchsize:(i+1)*batchsize,:,:]
    return batched_data

def data_batchlization_dnn(data,batchsize):
    # divide the dataset into batches, left out the final samples which not a full batch
    sample_num=data.shape[0]
    datadim=data.shape[1]
    batch_num=np.floor(sample_num/batchsize).astype('int')
    batched_data=np.zeros((batch_num,batchsize,datadim))
    for i in range(batch_num):
        batched_data[i,:,:]=data[i*batchsize:(i+1)*batchsize,:]
    return batched_data

def net_input_phase_encoding(data,T,t):
    # data.shape=(batch_size,datadim,channel)
    batch_size=data.shape[0]
    datadim=data.shape[1]
    channel=data.shape[2]

    # phase encoding
    W=datadim
    w=np.pi/W
    k=np.pi/T
    x=np.arange(W)
    x=np.reshape(x,(1,datadim))
    y_encoding = np.sin(w * x + k * t)
    y_encoding = np.reshape(y_encoding, (batch_size, datadim, 1))
    # if t is a vector with shape=(batch_size,1)
    # then broadcast to y_encoding.shape=(batch_size,datadim,1)

    # concat as the network input
    net_input=np.concatenate([data,y_encoding],axis=2)
    return net_input

def padding_tile(padding_data,padding_size):
    # padding_size is the half of the shape difference of the model
    # padding_data shape=(batch_size,datadim,channel)
    batch_size=padding_data.shape[0]
    datadim=padding_data.shape[1]
    channel=padding_data.shape[2]
    padded_data=np.zeros((batch_size,int(datadim+2*padding_size),channel))
    for i in range(batch_size):
        data_flip = np.flipud(padding_data[i,:,:])
        mid=np.concatenate([data_flip,padding_data[i,:,:],data_flip],axis=0)
        padded_data[i,:,:]=mid[int(datadim-padding_size):int(2*datadim+padding_size),:]
    return padded_data

def time_bar_encoding(input_ECG):
    # input_ECG.shape=(length,1)
    # R peak detection
    fs=360
    amp=0.3
    detectors = Detectors(fs)
    R_peak_index=detectors.swt_detector(input_ECG[:,0])
    R_peak_index=np.array(R_peak_index)
    # cos insert
    peak_num=R_peak_index.shape[0]
    if R_peak_index[0]>0:
        first_inter=R_peak_index[1]-R_peak_index[0]
        time_bar_insert=np.arange(first_inter)
        w=2*np.pi/first_inter
        time_bar_insert=amp*np.cos(w*time_bar_insert)
        time_bar_insert=time_bar_insert[(first_inter-R_peak_index[0]):]
        time_bar=time_bar_insert
    else:
        time_bar=np.ones((1,1))
    for i in range(peak_num-1):
        RR_inter=R_peak_index[i+1]-R_peak_index[i]
        time_bar_insert=np.arange(RR_inter)
        w=2*np.pi/RR_inter
        time_bar_insert=amp*np.cos(w*time_bar_insert)
        time_bar=np.concatenate([time_bar,time_bar_insert],axis=0)
    time_bar=np.reshape(time_bar,(time_bar.shape[0],1))
    return time_bar