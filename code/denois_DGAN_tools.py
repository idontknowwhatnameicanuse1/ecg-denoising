import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, \
    GlobalAveragePooling1D, MaxPooling1D
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
    noise = (Data_noise_BW * mix_por[0] + Data_noise_EM * mix_por[1] + Data_noise_MA * mix_por[2]) / np.sum(mix_por)
    noise = noise[:Data_clean_ECG.shape[0], :]
    noise = noise  # 1.2
    # noise=np.random.normal(size=(Data_clean_ECG.shape[0],Data_clean_ECG.shape[1]))
    # SNR adjust
    k = (np.sqrt(np.sum(Data_clean_ECG ** 2) / (np.sum(noise ** 2) * np.power(10, (SNR / 10))))).astype('double')
    # match data gen
    Data_noisy = Data_clean_ECG + k * noise
    Data_noisy = (Data_noisy - np.min(Data_noisy)) / \
                 (np.max(Data_noisy) - np.min(Data_noisy))
    Data_clean_ECG = (Data_clean_ECG - np.min(Data_clean_ECG)) / \
                     (np.max(Data_clean_ECG) - np.min(Data_clean_ECG))
    # baseline=np.real(np.fft.fftshift(np.fft.fft(Data_clean_ECG))[0,0])
    # Data_clean_ECG=Data_clean_ECG-baseline

    # Data_noisy=(Data_noisy-np.mean(Data_noisy))/np.std(Data_noisy)
    # Data_clean_ECG=(Data_clean_ECG-np.mean(Data_clean_ECG))/(3*np.std(Data_clean_ECG))

    match_sample_long = np.concatenate((Data_clean_ECG, Data_noisy), axis=1)
    return match_sample_long


def longdata_div(long_data, div_size):
    # divide the long match data into pieces
    T = long_data.shape[0]
    piece_num = (np.floor(T / div_size)).astype('int')
    match_sample = np.zeros((piece_num, div_size, 2))
    recover_bar = np.zeros((piece_num, 1))
    for i in range(piece_num):
        x0 = long_data[i * div_size:(i + 1) * div_size, 0]
        x1 = long_data[i * div_size:(i + 1) * div_size, 1]
        x0 = np.reshape(x0, (div_size, 1))
        x1 = np.reshape(x1, (div_size, 1))

        recover_bar[i, 0] = np.real((1 / div_size * np.fft.fft(x0, axis=0))[0])
        x0 = x0 - np.real((1 / div_size * np.fft.fft(x0, axis=0))[0])
        match_sample[i, :, 0] = np.reshape(x0, (div_size))

        x1 = x1 - np.real((1 / div_size * np.fft.fft(x1, axis=0))[0])
        match_sample[i, :, 1] = np.reshape(x1, (div_size))
        # plt.plot(x0)
        # plt.show()
        # plt.plot(x1)
        # plt.show()
    return match_sample, recover_bar


def longdata_div_multi_channel(long_data, div_size):
    # divide the long match data into pieces
    T = long_data.shape[0]
    extra_windows = 3
    piece_num = (np.floor(T / div_size)).astype('int')
    match_sample = np.zeros((piece_num - extra_windows, div_size, 2 + extra_windows))
    for i in range(extra_windows, piece_num):
        k = i - extra_windows
        match_sample[k, :, :2] = long_data[i * div_size:(i + 1) * div_size, :]
        extra_noisy_signal = np.reshape(long_data[(i - extra_windows) * div_size:i * div_size, 1]
                                        , (div_size, extra_windows))
        match_sample[k, :, 2:2 + extra_windows] = extra_noisy_signal
    return match_sample


def mit_data_read_gen(div_size, SNR, mix_por=np.array([1, 0, 0])):
    # data path
    # DataDir_clean_ECG ="/home/zhanghaowei/project/denoiseCGAN/mit_ECG_csv/"
    # DataDir_noise_BW = "/home/zhanghaowei/project/denoiseCGAN/mit_noise_csv/"
    # DataDir_noise_EM = "/home/zhanghaowei/project/denoiseCGAN/mit_noise_csv/"
    # DataDir_noise_MA = "/home/zhanghaowei/project/denoiseCGAN/mit_noise_csv/"

    DataDir_clean_ECG = "/home/uu201912628/project/mit_ECG_csv/"
    DataDir_noise_BW = "/home/uu201912628/project/mit_noise_csv/"
    DataDir_noise_EM = "/home/uu201912628/project/mit_noise_csv/"
    DataDir_noise_MA = "/home/uu201912628/project/mit_noise_csv/"

    # DataDir_clean_ECG ="/root/denois_CGAN/mit_ECG_csv/"
    # DataDir_noise_BW = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_EM = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_MA = "/root/denois_CGAN/mit_noise_csv/"

    # noise reading
    Data_noise_BW = pd.read_csv((DataDir_noise_BW + 'bw.csv'))
    Data_noise_EM = pd.read_csv((DataDir_noise_EM + 'em.csv'))
    Data_noise_MA = pd.read_csv((DataDir_noise_MA + 'ma.csv'))
    Data_noise_BW = Data_noise_BW.to_numpy()
    Data_noise_EM = Data_noise_EM.to_numpy()
    Data_noise_MA = Data_noise_MA.to_numpy()

    # clean ECG reading & match data generation
    match_sample = []
    filenames = os.listdir(DataDir_clean_ECG)
    nfile = len(filenames)
    for i in range(nfile):
        fulname = (DataDir_clean_ECG + filenames[i])
        Data_clean_ECG = pd.read_csv(fulname)
        Data_clean_ECG = Data_clean_ECG.to_numpy()

        # randomly choose a col in noise data as noise
        col_rand_array = np.arange(Data_noise_BW.shape[1])
        np.random.shuffle(col_rand_array)
        BW = Data_noise_BW[:, col_rand_array[0]]
        # BW = Data_noise_BW[:, 0]
        BW = np.reshape(BW, (Data_clean_ECG.shape[0], 1))

        col_rand_array = np.arange(Data_noise_EM.shape[1])
        np.random.shuffle(col_rand_array)
        EM = Data_noise_EM[:, col_rand_array[0]]
        # EM = Data_noise_EM[:, 0]
        EM = np.reshape(EM, (Data_clean_ECG.shape[0], 1))

        col_rand_array = np.arange(Data_noise_MA.shape[1])
        np.random.shuffle(col_rand_array)
        MA = Data_noise_MA[:, col_rand_array[0]]
        # MA = Data_noise_MA[:, 0]
        MA = np.reshape(MA, (Data_clean_ECG.shape[0], 1))

        # match data generation
        match_sample_long = match_Datagen(Data_clean_ECG, BW, EM, MA,
                                          SNR, mix_por)
        sample = longdata_div(match_sample_long, div_size)  # match_sample.shape=(sample_num,time,type)
        if match_sample == []:
            match_sample = sample
        else:
            match_sample = np.concatenate((match_sample, sample), axis=0)
    return match_sample


def data_batchlization(data, batchsize):
    # divide the dataset into batches, left out the final samples which not a full batch
    sample_num = data.shape[0]
    datadim = data.shape[1]
    channel = data.shape[2]
    batch_num = np.floor(sample_num / batchsize).astype('int')
    batched_data = np.zeros((batch_num, batchsize, datadim, channel))
    for i in range(batch_num):
        batched_data[i, :, :, :] = data[i * batchsize:(i + 1) * batchsize, :, :]
    return batched_data


def data_batchlization_gen_noise_try(data, batchsize, SNR):
    # divide the dataset into batches, left out the final samples which not a full batch
    sample_num = data.shape[0]
    datadim = data.shape[1]
    channel = data.shape[2]
    batch_num = np.floor(sample_num / batchsize).astype('int')
    # gen noise
    gen_core = tf.keras.models.load_model('/root/autodl-tmp/project/gen_noise_core/G')

    batched_data = np.zeros((batch_num, batchsize, datadim, channel))
    for i in range(batch_num):
        batched_data[i, :, :, 0] = data[i * batchsize:(i + 1) * batchsize, :, 0]
        Data_clean_ECG = batched_data[i, :, :, 0]
        Data_clean_ECG = np.reshape(Data_clean_ECG, (batchsize, datadim, 1))
        gauss = np.random.normal(0, 1, size=(batchsize, datadim, 1))
        gen_noise_input = np.concatenate([Data_clean_ECG, gauss], axis=2)
        noise = gen_core.predict(gen_noise_input)

        k = (np.sqrt(np.sum(Data_clean_ECG ** 2) / (np.sum(noise ** 2) * np.power(10, (SNR / 10))))).astype('double')
        Data_noisy = Data_clean_ECG + k * noise
        Data_noisy = (Data_noisy - np.min(Data_noisy)) \
                     / (np.max(Data_noisy) - np.min(Data_noisy))
        batched_data[i, :, :, 1] = np.squeeze(Data_noisy)
    return batched_data


def recording_norm(long_match_data):
    # normalization for every sample, using max and min norm
    length = long_match_data.shape[0]
    channel = long_match_data.shape[1]
    normed_data = np.zeros((length, channel))
    # normed_recov=np.zeros((sample_num,2))
    normed_reconv = 0
    clean_data = long_match_data[:, 0]
    noisy_data = long_match_data[:, 1]
    normed_data[:, 0] = 2 * (clean_data - np.min(clean_data)) / (
            np.max(clean_data) - np.min(clean_data)) - 1
    normed_data[:, 1] = 2 * (noisy_data - np.min(noisy_data)) / (
            np.max(noisy_data) - np.min(noisy_data)) - 1
    # normed_recov[i, 0] = np.min(noisy_sample)
    # normed_recov[i, 1] = np.max(noisy_sample)
    return normed_data, normed_reconv


def piece_norm(dataset):
    # normalization for every sample, using max and min norm
    sample_num = dataset.shape[0]
    datadim = dataset.shape[1]
    channel = dataset.shape[2]
    normed_sample = np.zeros((sample_num, datadim, channel))
    normed_recov = np.zeros((sample_num, 2))
    for i in range(sample_num):
        clean_sample = dataset[i, :, 0]
        normed_sample[i, :, 0] = (clean_sample - np.min(clean_sample)) \
                                 / (np.max(clean_sample) - np.min(clean_sample))
        # normed_sample[i,:,0]=clean_sample
        for j in range(channel):
            noisy_sample = dataset[i, :, j]
            normed_sample[i, :, j] = (noisy_sample - np.min(noisy_sample)) \
                                     / (np.max(noisy_sample) - np.min(noisy_sample))

        normed_recov[i, 0] = np.min(clean_sample)
        normed_recov[i, 1] = np.max(clean_sample)
    return normed_sample, normed_recov


def piece_denorm(normed_sample, normed_recov):
    sample_num = normed_sample.shape[0]
    datadim = normed_sample.shape[1]
    channel = normed_sample.shape[2]
    denormed_sample = np.zeros((sample_num, datadim, channel))
    for i in range(sample_num):
        min = normed_recov[i, 0]
        max = normed_recov[i, 1]
        denormed_sample[i, :, :] = min + denormed_sample[i, :, :] * (max - min)
    return denormed_sample


def batch_conv(input, kernel, padding='same'):
    num = input.shape[0]  # input.shape=(num,div_size,channel=1)
    div_size = input.shape[1]
    result = np.zeros((num, div_size))
    conv_kernel = np.squeeze(kernel)
    conv_input = np.squeeze(input)
    for i in range(num):
        result[i, :] = np.convolve(conv_input[i, :], conv_kernel, mode='same')
        # result[i,:]=np.convolve(result[i,:],conv_kernel,mode='same')
        # result[i, :] = np.convolve(result[i, :], conv_kernel, mode='same')
        # result[i, :] = np.convolve(result[i, :], conv_kernel, mode='same')
        # result[i,:] = np.exp(-100 * (result[i,:] - np.mean(result[i,:]))**2)
    result = np.reshape(result, (num, div_size, 1))
    # result=result-threshold
    # result[result<0]=0

    return result


def kernel_gen(kernel_size, sigma, amp):
    kernel = np.zeros((kernel_size, 1))
    miu = int(kernel_size / 2)
    for i in range(kernel_size):
        kernel[i, :] = amp * np.exp(-(i - miu) ** 2 / (2 * sigma ** 2))
    # for i in range(kernel_size):
    #     kernel[i,:]=amp-abs(2*amp/kernel_size*i-amp)
    return kernel


def dataset_gen(match_sample, recover_bar, train_por):
    # generate trainset and testset from total dataset
    num = match_sample.shape[0]
    train_num = np.floor(num * train_por).astype('int')

    # randomly choose sample from total data
    index_rand_array = np.arange(match_sample.shape[0])
    np.random.shuffle(index_rand_array)
    # trainset = match_sample[index_rand_array[:train_num],:,:]
    # testset=match_sample[index_rand_array[train_num:],:,:]
    trainset = match_sample[index_rand_array[:train_num]]
    testset = match_sample[index_rand_array[train_num:]]
    # recover_bar=recover_bar[index_rand_array]
    return trainset, testset


def dataset_gen_testset_valid(match_sample, recover_bar, train_por):
    # generate trainset and testset from total dataset
    num = match_sample.shape[0]
    train_num = np.floor(num * train_por).astype('int')

    # randomly choose sample from total data
    index_rand_array = np.arange(match_sample.shape[0])
    np.random.shuffle(index_rand_array)
    trainset = match_sample[index_rand_array[:train_num], :, :]
    testset = match_sample[index_rand_array[train_num:], :, :]
    recover_bar = recover_bar[index_rand_array, :]
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
def nfold_dataset_gen(div_size, SNR, train_por, mix_por=np.array([1, 0, 0])):
    # data path
    DataDir_clean_ECG = "/home/uu201912628/project/mit_ECG_csv/"
    DataDir_noise_BW = "/home/uu201912628/project/mit_noise_csv/"
    DataDir_noise_EM = "/home/uu201912628/project/mit_noise_csv/"
    DataDir_noise_MA = "/home/uu201912628/project/mit_noise_csv/"

    # DataDir_clean_ECG ="/root/denois_CGAN/mit_ECG_csv/"
    # DataDir_noise_BW = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_EM = "/root/denois_CGAN/mit_noise_csv/"
    # DataDir_noise_MA = "/root/denois_CGAN/mit_noise_csv/"

    # noise reading
    Data_noise_BW = pd.read_csv((DataDir_noise_BW + 'bw.csv'))
    Data_noise_EM = pd.read_csv((DataDir_noise_EM + 'em.csv'))
    Data_noise_MA = pd.read_csv((DataDir_noise_MA + 'ma.csv'))
    Data_noise_BW = Data_noise_BW.to_numpy()
    Data_noise_EM = Data_noise_EM.to_numpy()
    Data_noise_MA = Data_noise_MA.to_numpy()

    # clean ECG reading & match data generation
    match_sample = []
    recover_bar_total = []
    filenames = os.listdir(DataDir_clean_ECG)
    nfile = len(filenames)

    # fold control
    name_bank = ['selected_ECG_205_MLII.csv', 'selected_ECG_103_MLII.csv',
                 'selected_ECG_219_MLII.csv', 'selected_ECG_223_MLII.csv',
                 'selected_ECG_122_MLII.csv', 'selected_ECG_111_MLII.csv',
                 'selected_ECG_230_MLII.csv', 'selected_ECG_213_MLII.csv',
                 'selected_ECG_116_MLII.csv', 'selected_ECG_105_MLII.csv']

    # random select certain num of files
    # np.random.shuffle(filenames)

    BW = Data_noise_BW[:, 1]  # 1
    BW = np.reshape(BW, (BW.shape[0], 1))

    EM = Data_noise_EM[:, 1]  # 1
    EM = np.reshape(EM, (EM.shape[0], 1))

    MA = Data_noise_MA[:, 0]
    MA = np.reshape(MA, (MA.shape[0], 1))

    if_train = 1
    # trainset gen
    normed_match_sample = 0
    snr_bank = np.array([0, 1.25, 5])
    mix_choice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                           [1, 1, 1]])
    if if_train:
        for i in range(int(nfile * train_por)):
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
    trainset, leftout = dataset_gen(match_sample, recover_bar_total, train_por=1)
    # trainset=0

    # testset gen
    match_sample = []
    recover_bar_total = []
    normed_match_sample = 0
    # for i in range(int(nfile*train_por),int(nfile)):
    for i in range(int(nfile * train_por), int(nfile)):
        # filenames gen & clean data read
        fulname = (DataDir_clean_ECG + name_bank[i])
        Data_clean_ECG = pd.read_csv(fulname)
        Data_clean_ECG = Data_clean_ECG.to_numpy()
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
    testset, leftout, recover_bar_total = dataset_gen_testset_valid(match_sample, recover_bar_total, train_por=1)
    # testset=0
    return trainset, testset, recover_bar_total


def SNR(ground_truth, data):
    noise = data - ground_truth

    snr = 10 * np.log10((np.sum(ground_truth ** 2)) / (np.sum(noise ** 2)))
    return snr


def RMSE(ground_truth, data):
    rmse = np.sqrt(np.mean((ground_truth - data) ** 2))
    return rmse


def baseline_correction(ground_truth, data):
    batch_count = ground_truth.shape[0]
    batch_size = ground_truth.shape[1]
    datadim = ground_truth.shape[2]
    N = 512
    data_correction = np.zeros((batch_count, batch_size, datadim, 1))
    for j in range(batch_count):
        for i in range(batch_size):
            n = ground_truth[j, i, :, :] - data[j, i, :, :]
            correction = abs(np.fft.fftshift(np.fft.fft(n, N)))[0] * np.ones(
                (n.shape[0]))
            data_correction[j, i, :, :] = data[j, i, :, :] + correction
    return data_correction


def baseline_correction_single_batch(ground_truth, data):
    batch_size = ground_truth.shape[0]
    datadim = ground_truth.shape[1]
    N = 512
    data_correction = np.zeros((batch_size, datadim, 1))
    for i in range(batch_size):
        n = ground_truth[i, :, :] - data[i, :, :]
        correction = np.fft.fftshift(np.fft.fft(n, N))[0]  # * np.ones((n.shape[0]))
        correction = np.reshape(correction, (datadim, 1))
        data_correction[i, :, :] = data[i, :, :] + correction
    return data_correction


def piece_CC(ground_truth, data):
    # ground_truth and data.shape=(sample_num,div_size,1)
    ground_truth_long = np.reshape(ground_truth,
                                   (ground_truth.shape[0] * ground_truth.shape[1], 1))
    data_long = np.reshape(data, (data.shape[0] * data.shape[1], 1))
    cc = long_data_CC(ground_truth_long, data_long)
    return cc


def long_data_CC(ground_truth, data):
    # ground_truth and data.shape=(sample_point_num,1)
    # cor=np.corrcoef(ground_truth,data)
    g_ = ground_truth.mean()
    d_ = data.mean()

    cc = np.sum((ground_truth - g_) * (data - d_)) / \
         ((np.sqrt(np.sum((ground_truth - g_) ** 2))) * (np.sqrt(np.sum((data - d_) ** 2))))
    return cc


class SpectralNorm(tf.keras.constraints.Constraint):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter

    def call(self, input_weights):
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
        u = tf.random.normal((w.shape[0], 1))
        v = 0
        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a=True)
            v /= tf.norm(v)
            u = tf.matmul(w, v)
            u /= tf.norm(u)
        spec_norm = tf.matmul(u, tf.matmul(w, v), transpose_a=True)
        return input_weights / spec_norm


# gru upgrade
def create_generator():
    generator = Sequential()
    generator.add(keras.Input(shape=(310,)))
    generator.add(keras.layers.Dense(units=310, activation='tanh'))
    generator.add(keras.layers.Dense(units=250, activation='tanh'))
    generator.add(keras.layers.Dense(units=250, activation='tanh'))
    generator.add(keras.layers.Dense(units=250, activation='tanh'))
    generator.add(keras.layers.Dense(units=310, activation='sigmoid'))
    generator.compile()
    return generator


# over fit
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(keras.Input(shape=(310,)))
    discriminator.add(keras.layers.Dense(units=310, activation='tanh'))
    discriminator.add(keras.layers.Dense(units=150, activation='tanh'))
    discriminator.add(keras.layers.Dense(units=150, activation='tanh'))
    discriminator.add(keras.layers.Dense(units=1, activation='sigmoid'))
    discriminator.compile()
    return discriminator


def plot_epoch_denoised_signals(batched_trainset_noisy, batched_trainset_clean, batched_testset_noisy,
                                batched_testset_clean, epoch, generator, batch_count):
    # rand=int(np.random.randint(0,batch_count,1))
    rand = 2
    batch_size = batched_trainset_clean.shape[1]
    datadim = batched_trainset_clean.shape[2]
    channel = batched_trainset_clean.shape[3]
    noisy_input = batched_trainset_noisy
    ground_truth = batched_trainset_clean

    test_ground_truth = batched_testset_clean
    test_noisy_input = batched_testset_noisy
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
    test_denoise_signal = generator.predict(test_noisy_input)
    # test_denoise_signal=baseline_correction_single_batch(test_ground_truth, test_denoise_signal)
    # for j in range(batch_size):
    #     test_denoise_signal[j,:,:]=(test_denoise_signal[j,:,:]-np.min(test_denoise_signal[j,:,:]))/\
    #                            (np.max(test_denoise_signal[j,:,:])-np.min(test_denoise_signal[j,:,:]))

    rand2 = 8
    # select a piece to show
    ground_truth_plot = ground_truth[rand2, 0, :, 0]
    noisy_input_plot = noisy_input[rand2, 0, :, 0]
    denoised_signal_plot = denoised_signal[0, :, 0]

    test_ground_truth_plot = test_ground_truth[0, 0, :, 1]
    test_noisy_input_plot = test_noisy_input[0, 0, :, 1]
    test_denoise_signal_plot = test_denoise_signal[0, :, 1]

    # performance evaluation
    train_snr = SNR(ground_truth, denoised_signal)
    train_rmse = RMSE(ground_truth, denoised_signal)
    valid_snr = SNR(test_ground_truth, test_denoise_signal)
    valid_rmse = RMSE(test_ground_truth, test_denoise_signal)

    # result=np.asarray([train_snr,train_rmse])
    # pd.DataFrame(result).to_csv(('/home/zhanghaowei/project/denoiseCGAN/train_process/train_process_'+str(epoch)+'.csv'))
    # pd.DataFrame(result).to_csv(
    #     ('/home/uu201912628/project/train_process/train_process_' + str(epoch) + '.csv'))
    # pd.DataFrame(result).to_csv(
    #     ('/root/denois_CGAN/train_process/train_process_' + str(epoch) + '.csv'))

    print("epoch", epoch, " training, snr=", train_snr, " rmse=", train_rmse)
    print("epoch", epoch, " validing, snr=", valid_snr, " rmse=", valid_rmse)

    N = 1536
    # ground_truth_plot_baseline=np.abs(np.fft.fft(ground_truth_plot,axis=0))[0]*np.ones((ground_truth_plot.shape[0]))
    # denoised_signal_plot_baseline=np.abs(np.fft.fft(denoised_signal_plot,axis=0))[0,0]*np.ones((denoised_signal_plot.shape[0]))
    # noisy_input_plot_baseline=np.abs(np.fft.fft(noisy_input_plot,axis=0))[0]*np.ones((noisy_input_plot.shape[0]))

    # test_ground_truth_plot_baseline=np.abs(np.fft.fft(test_ground_truth_plot,axis=0))[0]*np.ones((test_ground_truth_plot.shape[0]))
    # test_denoise_signal_plot_baseline=np.abs(np.fft.fft(test_denoise_signal_plot,axis=0))[0,0]*np.ones((test_denoise_signal_plot.shape[0]))
    # test_noisy_input_plot_baseline=np.abs(np.fft.fft(test_noisy_input_plot,axis=0))[0]*np.ones((test_noisy_input_plot.shape[0]))

    print(("epoch " + str(epoch) + "training denoise plot"))
    plt.subplot(221)
    x = range(len(ground_truth_plot))
    plt.plot(x, ground_truth_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('ground truth')
    plt.subplot(222)
    plt.plot(x, noisy_input_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('noisy signal')
    plt.subplot(223)
    x = range(len(denoised_signal_plot))
    plt.plot(x, denoised_signal_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('denoise signal')
    plt.subplot(224)
    x = range(len(ground_truth_plot))
    plt.plot(x, ground_truth_plot, x, denoised_signal_plot)
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
    print(("epoch " + str(epoch) + "validing denoise plot"))
    plt.subplot(221)
    x = range(len(test_ground_truth_plot))
    plt.plot(x, test_ground_truth_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('ground truth')
    plt.subplot(222)
    plt.plot(x, test_noisy_input_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('noisy signal')
    plt.subplot(223)
    x = range(len(test_denoise_signal_plot))
    plt.plot(x, test_denoise_signal_plot)
    # plt.ylim([-0.1,1.1])
    plt.title('denoise signal')
    plt.subplot(224)
    x = range(len(test_ground_truth_plot))
    plt.plot(x, test_ground_truth_plot, x, test_denoise_signal_plot)
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


class DNNCGAN(Model):
    def __init__(self):
        super(DNNCGAN, self).__init__()

        self.layer_1 = create_generator()
        self.layer_2 = create_discriminator()

    def G_loss(self, inputs, raw_data, epoch):
        g_output = self.layer_1(inputs)
        # D_input = tf.concat([g_output, inputs], 2)
        output = self.layer_2(g_output)

        base = tf.reduce_sum((output - 1) ** 2)
        # if epoch<3000:
        # if epoch<150:
        # if epoch<150:
        #     lumbda0=0# 1e-3
        # else:

        lumbda0 = 1  # 1
        lumbda1 = 0.7  # 0.8      4
        lumbda2 = 0.2  # 0.2  # 0.3     90
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
        ldist = tf.sqrt(tf.reduce_sum((g_output - raw_data) ** 2))
        lmax = tf.reduce_max(abs(g_output - raw_data))
        # cc=piece_CC(raw_data,g_output)
        # print('cc=',cc)
        N = 512
        batch_size = 128
        div_size = 512
        # baseline_loss=np.zeros(batch_size)
        # for i in range(batch_size):
        #     x=g_output.numpy()[i,:,:]-raw_data[i,:,:]
        #     baseline_loss[i] = abs(np.fft.fftshift(np.fft.fft(x)))[0,0]
        # baseline_loss=np.sum(baseline_loss)
        #
        # baseline_loss = tf.cast(baseline_loss, dtype=tf.float32)
        print("Gloss base=", base, " ldist=", ldist, " lmax=", lmax)
        # print("baseline loss=",baseline_loss)
        loss = lumbda0 * base + lumbda1 * ldist + lumbda2 * lmax

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
        real_output = self.layer_2(real_input)

        rand_label_collid = 0.05 * np.random.normal(1)

        base = tf.reduce_sum((gen_output - 0.1 + rand_label_collid) ** 2
                             + (real_output - 0.9 + rand_label_collid) ** 2) / 2
        lamuda = 1
        loss = base * lamuda
        print("\nDloss loss=", loss)
        # D_loss=np.asarray([loss])
        # pd.DataFrame(D_loss).to_csv(
        #     ('/root/autodl-tmp/project/train_process/D_loss_history'+str(epoch)+'.csv'))
        return loss

    def G_get_grad(self, input, raw_data, epoch):
        with tf.GradientTape() as tape:
            tape.watch(self.layer_1.variables)
            L = self.G_loss(input, raw_data, epoch)
            g = tape.gradient(L, self.layer_1.variables)
        return g

    def D_get_grad(self, gen_input, real_input, epoch):
        with tf.GradientTape() as tape:
            tape.watch(self.layer_2.variables)
            L = self.D_loss(gen_input, real_input, epoch)
            g = tape.gradient(L, self.layer_2.variables)
        return g

    def training(self, input, testset, epochs, lr_G, lr_D, batch_size=256):
        # normed_trainset=input
        # every sample normalization
        # normed_trainset,leftout=piece_norm(input)

        # normed_testset,leftout=piece_norm(testset)

        normed_trainset = input  # trainset
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
        generator = self.layer_1
        discriminator = self.layer_2
        gan = self
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
            if e == 12:
                lr_G = 0.1 * lr_G
                lr_D = 0.5 * lr_D
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

                training_strategy = 1  # training_stratege is to set
                # 1:mix training(mix pos & neg sampleto train D) or
                # 0:class training(random select a class, pos or neg to train D each batch)
                if training_strategy:
                    # randomly choose a combination as the input of D
                    mix = np.concatenate([X_real_noisy, X_gen_noisy], axis=0)
                    mix_label = np.ones(2 * batch_size)  # -0.1*np.random.randint(0,1,1)
                    mix_label[batch_size:] = 0  # +0.1*np.random.randint(0,1,1)

                    rand_index = np.arange(2 * batch_size)
                    np.random.shuffle(rand_index)
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
                self.layer_1.trainable = False
                self.layer_2.trainable = True

                if True:  # e<5: #%1==0: # and e>=3:
                    grad_D = self.D_get_grad(X_gen_noisy, X_real_noisy, epoch=_)
                    tf.keras.optimizers.RMSprop(lr_D).apply_gradients(zip(grad_D, self.layer_2.variables))

                # D valid
                y_pred = np.squeeze(self.layer_2.predict(X, batch_size=128))
                y_pred[y_pred > 0.5] = 1  # 0.9
                y_pred[y_pred < 0.5] = 0  # 0.1
                if training_strategy:
                    y_dis[y_dis > 0.5] = 1
                    y_dis[y_dis < 0.5] = 0
                else:
                    y_dis = y_dis * np.ones(batch_size)
                acc = np.sum(y_dis == y_pred) / (2 * batch_size)
                # acc = (y_dis == y_pred).astype(float).mean()
                print("acc=", acc)

                if True:  # e<=5 or e%3==0: # or (e==1 and _<=15):
                    # input the denoise combination into fixed D and get the output of D
                    noisy_gan_input = noisy_ECG
                    y_gen_train = np.ones(batch_size)

                    # fix the discriminator when training GAN
                    self.layer_2.trainable = False
                    self.layer_1.trainable = True

                    # GAN train
                    grad_G = self.G_get_grad(noisy_gan_input, raw_data=clean_ECG, epoch=_)
                    tf.keras.optimizers.RMSprop(learning_rate=lr_G).apply_gradients(zip(grad_G, self.layer_1.variables))
            snr = plot_epoch_denoised_signals(batched_trainset, batched_testset, e, generator, batch_count)

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
        generator.save('/home/zhanghaowei/DDPM/saved_model/dnn_gan')

    def training_ddpm_adapt(self, train_clean, train_noisy, epochs, lr_G, lr_D, batch_size=256):
        # normed_trainset=input
        # every sample normalization
        # normed_trainset,leftout=piece_norm(input)

        # normed_testset,leftout=piece_norm(testset)
        batch_count = train_clean.shape[0]

        # CGAN definition
        generator = self.layer_1
        discriminator = self.layer_2
        gan = self
        self.compile()
        # gan = create_gan(discriminator, generator)

        # CGAN training
        for e in range(1, epochs + 1):
            print("Epoch %d" % e)
            if e == 12:
                lr_G = 0.1 * lr_G
                lr_D = 0.5 * lr_D
            for _ in tqdm(range(int(batch_count))):
                print('batch num=', _)
                noisedata_this_batch = train_noisy[_, :, :]
                # noisedata_this_batch=np.reshape(noisedata_this_batch,(batch_size,datadim,channel))
                cleandata_this_batch = train_clean[_, :, :]
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

                training_strategy = 1  # training_stratege is to set
                # 1:mix training(mix pos & neg sampleto train D) or
                # 0:class training(random select a class, pos or neg to train D each batch)
                if training_strategy:
                    # randomly choose a combination as the input of D
                    mix = np.concatenate([X_real_noisy, X_gen_noisy], axis=0)
                    mix_label = np.ones(2 * batch_size)  # -0.1*np.random.randint(0,1,1)
                    mix_label[batch_size:] = 0  # +0.1*np.random.randint(0,1,1)

                    rand_index = np.arange(2 * batch_size)
                    np.random.shuffle(rand_index)
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
                self.layer_1.trainable = False
                self.layer_2.trainable = True

                if True:  # e<5: #%1==0: # and e>=3:
                    grad_D = self.D_get_grad(X_gen_noisy, X_real_noisy, epoch=_)
                    tf.keras.optimizers.RMSprop(lr_D).apply_gradients(zip(grad_D, self.layer_2.variables))

                # D valid
                y_pred = np.squeeze(self.layer_2.predict(X, batch_size=128))
                y_pred[y_pred > 0.5] = 1  # 0.9
                y_pred[y_pred < 0.5] = 0  # 0.1
                if training_strategy:
                    y_dis[y_dis > 0.5] = 1
                    y_dis[y_dis < 0.5] = 0
                else:
                    y_dis = y_dis * np.ones(batch_size)
                acc = np.sum(y_dis == y_pred) / (2 * batch_size)
                # acc = (y_dis == y_pred).astype(float).mean()
                print("acc=", acc)

                if True:  # e<=5 or e%3==0: # or (e==1 and _<=15):
                    # input the denoise combination into fixed D and get the output of D
                    noisy_gan_input = noisy_ECG
                    y_gen_train = np.ones(batch_size)

                    # fix the discriminator when training GAN
                    self.layer_2.trainable = False
                    self.layer_1.trainable = True

                    # GAN train
                    grad_G = self.G_get_grad(noisy_gan_input, raw_data=clean_ECG, epoch=_)
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
        generator.save('/home/zhanghaowei/DDPM/saved_model/dnn_gan')


def denoise_CGAN_denoise(noisy_data, recover_bar, batch_size=256):
    batch_size = noisy_data.shape[1]
    datadim = noisy_data.shape[2]
    channel = noisy_data.shape[3]

    # batch count
    batch_num = noisy_data.shape[0]  # int(sample_num/batch_size)

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
    denoise_data = np.zeros((batch_num, batch_size, datadim, 12))

    # denoising
    for i in range(batch_num):
        # for k in range(recurrent_valid):
        denoise_data[i, :, :, :] = denoise_model.predict(noisy_data[i, :, :, :])
        # for j in range(batch_size):
        #     denoise_data[i,j,:,:]=denoise_data[i,j,:,:]+recover_bar[i*batch_size+j,0]

    return denoise_data


def denoise_DNN_CGAN_denoise(noisy_data,batch_size=256):
    batch_size = noisy_data.shape[1]
    datadim = noisy_data.shape[2]

    # batch count
    batch_num = noisy_data.shape[0]  # int(sample_num/batch_size)

    # trained model load
    # denoise_model=tf.keras.models.load_model('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
    # denoise_model = tf.keras.models.load_model('/root/autodl-tmp/project/denoise_core/denoise_G')
    # denoise_model = tf.keras.models.load_model('/home/uu201912628/project/denoise_core/denoise_G')
    # denoise_model = tf.keras.models.load_model('/root/denois_CGAN/denoise_core/denoise_G')
    denoise_model = tf.keras.models.load_model('/home/zhanghaowei/DDPM/saved_model/dnn_gan')

    # data batchilization
    # batched_testset = data_batchlization(noisy_data, batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # model predict for noise data
    # noise_data=batched_testset[:,:,:,1]
    # noise_data=np.reshape(noise_data,(batch_num,batch_size,datadim,1))

    # initialization
    # noisy_data = noisy_data[:, :, :, 0]
    denoise_data = np.zeros((batch_num, batch_size, datadim))

    # denoising
    # for i in range(batch_num):
    #     denoise_data[i, :, :] = denoise_model.predict(noisy_data[i, :, :])
    for j in range(noisy_data.shape[-1]):
        denoise_data[:, :, j]=denoise_model.predict(noisy_data[:, :, j])

    return denoise_data


def denoise_CGAN_valid(testset, recover_bar, batch_size=256):
    sample_num = testset.shape[0]
    datadim = testset.shape[1]
    channel = testset.shape[2]

    # batch count
    batch_num = int(sample_num / batch_size)

    # trained model load
    # denoise_model=tf.keras.models.load_model('/home/zhanghaowei/project/denoiseCGAN/denoise_core_epoch/denoise_G')
    # denoise_model = tf.keras.models.load_model('/root/autodl-tmp/project/denoise_core/denoise_G')
    denoise_model = tf.keras.models.load_model('/home/uu201912628/project/denoise_core/denoise_G')

    # sample normalization
    # normed_testset,normed_recov_bar=piece_norm(testset)
    # normed_testset=testset

    # data batchilization
    batched_testset = data_batchlization(testset,
                                         batch_size)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # batched_testset = data_batchlization_gen_noise_try(normed_testset,
    #                                                     batch_size,
    #                                                     SNR=0)  # batched_data.shape=(batch_index,batchsize,datadim,channel)

    # model predict for noise data
    noise_data = batched_testset[:, :, :, 1]

    # label
    ground_truth = batched_testset[:, :, :, 0]

    noise_data = np.reshape(noise_data, (batch_num, batch_size, datadim, 1))

    # initialization
    denoise_data = np.zeros((batch_num, batch_size, datadim, 1))

    # denoising
    recurrent_valid = 1
    for i in range(batch_num):
        # for k in range(recurrent_valid):
        denoise_data[i, :, :, :] = denoise_model.predict(noise_data[i, :, :, :])
        for j in range(batch_size):
            denoise_data[i, j, :, :] = denoise_data[i, j, :, :] + recover_bar[i * batch_size + j, 0]
            ground_truth[i, j, :] = ground_truth[i, j, :] + recover_bar[i * batch_size + j]
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
    ground_truth = np.reshape(ground_truth, (ground_truth.shape[0] * ground_truth.shape[1] * ground_truth.shape[2], 1))
    noise_data_plot = np.reshape(noise_data, (noise_data.shape[0] * noise_data.shape[1] * noise_data.shape[2], 1))
    denoise_data = np.reshape(denoise_data, (denoise_data.shape[0] * denoise_data.shape[1] * denoise_data.shape[2], 1))

    # reshape plot
    time_s = 182500
    time_e = 184500
    ground_truth_plot = ground_truth[time_s:time_e]
    noise_data_plot = noise_data_plot[time_s:time_e]
    denoise_data_plot = denoise_data[time_s:time_e]

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
    snr = SNR(ground_truth, denoise_data)
    rmse = RMSE(ground_truth, denoise_data)
    # snr = SNR(ground_truth, noise_data_plot)
    # rmse = RMSE(ground_truth, noise_data_plot)
    result = np.asarray([snr, rmse])
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
    return snr, rmse


