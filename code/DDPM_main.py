import numpy as np
import pandas as pd
import tensorflow as tf
import metrics
import inhouse_ecg_read as ier

import DDPM_tools as tls
import pre_process as sp
import plot_tools as ptls
import denoise_conditional_sampling_tools as dcst
import os
import matplotlib.pyplot as plt
import soft_threshold_WT_tools as stwt

import denois_RGAN_tools as gan_tls
import denois_DGAN_tools as dnngan_tls
import IDAE_tools as idae

# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']='1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

div_size= 1024 # 1024  # 360hz with 1024
SNR=0
training_por=0.8
mix_por=np.array([1,1,1])
path_name=tls.file_path(servier='hust_exp')

T=100 # 100
datadim=1024
batch_size=128 # 128
channel=12
beta_t=np.linspace(start=1e-4,stop=0.02,num=T) #1/(T-np.arange(T)+1)
# beta_t=tls.cosine_schedule(T=T,s=0.08)
sigma_t=np.sqrt(beta_t)

trainset,testset,recover_bar=sp.nfold_dataset_gen(path_name,div_size,SNR,training_por,if_train=1,mix_por=mix_por)

# trainset_clean=trainset[:,:,:12]
# trainset_clean=np.reshape(trainset_clean,(trainset_clean.shape[0]*trainset_clean.shape[-1],trainset_clean.shape[1]))
# trainset_clean=sp.data_batchlization_dnn(trainset_clean,batchsize=batch_size)
# trainset_noisy=trainset[:,:,12:]
# trainset_noisy=np.reshape(trainset_noisy,(trainset_noisy.shape[0]*trainset_noisy.shape[-1],trainset_noisy.shape[1]))
# trainset_noisy=sp.data_batchlization_dnn(trainset_noisy,batchsize=batch_size)
# 
# testset_noisy=testset[:,:,12:]
# testset_noisy=sp.data_batchlization(testset_noisy,batchsize=batch_size)[2:4]
# testset_clean=testset[:,:,:12]
# testset_clean=sp.data_batchlization(testset_clean,batchsize=batch_size)[2:4]

trainset_clean=np.reshape(trainset[:,:,0],(trainset.shape[0],trainset.shape[1],1))
trainset_clean=sp.data_batchlization(trainset_clean,batchsize=batch_size)
trainset_noisy=np.reshape(trainset[:,:,1],(trainset.shape[0],trainset.shape[1],1))
trainset_noisy=sp.data_batchlization(trainset_noisy,batchsize=batch_size)

testset_noisy=np.reshape(testset[:,:,1],(testset.shape[0],testset.shape[1],1))
testset_noisy=sp.data_batchlization(testset_noisy,batchsize=batch_size)
testset_clean=np.reshape(testset[:,:,0],(testset.shape[0],testset.shape[1],1))
testset_clean=sp.data_batchlization(testset_clean,batchsize=batch_size)

DDPM = tls.DDPM_model()

# ddpm train
# DDPM.train(path_name=path_name, trainset=trainset_clean, epochs=5, beta_t=beta_t, lr=1e-4)

# encoder train
# dcst.encoder_model_train(clean_data=trainset_clean,noisy_data=trainset_noisy,beta_t=beta_t,batch_size=batch_size,path_name=path_name)

denoise_data=tls.denoise_conditional_sampling(condition=testset_noisy,ground=testset_clean,path_name=path_name,
                          beta_t=beta_t,sigma_t=sigma_t,
                          datadim=datadim,batch_size=batch_size,channel=channel)

# denoise_data = gan_tls.denoise_CGAN_denoise(noisy_data=testset_noisy, recover_bar=recover_bar,
#                                             batch_size=128)

# post process
batch_num=denoise_data.shape[0]
RMSE_piece=np.zeros(batch_num*batch_size)
for i in range(batch_num):
    for j in range(batch_size):
        denoise_data[i, j, :, :] = denoise_data[i, j, :, :] + recover_bar[i * batch_size + j]
        testset_clean[i, j, :] = testset_clean[i, j, :] + recover_bar[i * batch_size + j]
        testset_noisy[i, j, :] = testset_noisy[i, j, :] + recover_bar[i * batch_size + j]
        RMSE_piece[i*batch_size+j] = metrics.RMSE(denoise_data[i,j,:,:], testset_clean[i,j,:])

plt.plot(RMSE_piece)
RMSE=metrics.RMSE(denoise_data,testset_clean)
SNR=metrics.SNR(denoise_data,testset_clean)
print("RMSE=",RMSE)
print("SNR=",SNR)

# plot
clean_data=testset_clean[:1,:,:,:]
noisy_data=testset_noisy[:1,:,:,:]
sample_num=5
ptls.show_img(clean_data,noisy_data,denoise_data,sample_num)
st=41000  # 18000   103:41000  # plot window use: record 116 :time 41000
et=st+1024*2
ptls.demo_plot(path_name=path_name,ground_truth=testset_clean,noise_data=testset_noisy,denoise_data=denoise_data,
               start_time=st,end_time=et)

ptls.compare_demo_plot(path_name,ground_truth=clean_data,noise_data=noisy_data,start_time=st,end_time=et,
                      beta_t=beta_t,sigma_t=sigma_t,datadim=datadim,batch_size=batch_size,channel=channel,recover_bar=recover_bar)

# fold valid
state = False
if state:
    snr_bank = np.array([0, 1.25, 5])
    mix_choice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                           [1, 1, 1]])
    snr = np.zeros((snr_bank.shape[0], mix_choice.shape[0]))
    rmse = np.zeros((snr_bank.shape[0], mix_choice.shape[0]))
    for j in range(snr_bank.shape[0]):
        for k in range(7):
            print('combination ', 7*j + k + 2)
            SNR = snr_bank[j]
            mix_por = mix_choice[k, :]
            trainset, testset, recover_bar = sp.nfold_dataset_gen(path_name,div_size, SNR, training_por,if_train=0, mix_por=mix_por)
            # trainset, testset, recover_bar = ier.inhouse_dataset_gen(path_name, div_size, if_train=0, test_snr=SNR,
            #                                                          mix_por=mix_por)
            # public
            # trainset_clean = np.reshape(trainset[:, :, 0], (trainset.shape[0], trainset.shape[1], 1))
            # trainset_clean = sp.data_batchlization(trainset_clean, batchsize=batch_size)
            # trainset_noisy = np.reshape(trainset[:, :, 1], (trainset.shape[0], trainset.shape[1], 1))
            # trainset_noisy = sp.data_batchlization(trainset_noisy, batchsize=batch_size)
            #
            testset_noisy = np.reshape(testset[:, :, 1], (testset.shape[0], testset.shape[1], 1))
            testset_noisy = sp.data_batchlization(testset_noisy, batchsize=batch_size)
            testset_clean = np.reshape(testset[:, :, 0], (testset.shape[0], testset.shape[1], 1))
            testset_clean = sp.data_batchlization(testset_clean, batchsize=batch_size)

            # in-house
            # testset_noisy = testset[:, :, 12:]
            # testset_noisy = sp.data_batchlization(testset_noisy, batchsize=batch_size)
            # testset_clean = testset[:, :, :12]
            # testset_clean = sp.data_batchlization(testset_clean, batchsize=batch_size)

            # ddpm denoise
            # denoise_data = tls.denoise_conditional_sampling(condition=testset_noisy, factor=0.8,
            #                                                 path_name=path_name,
            #                                                 beta_t=beta_t, sigma_t=sigma_t,
            #                                                 datadim=datadim, batch_size=batch_size, channel=channel)

            # cgan denoise
            denoise_data = gan_tls.denoise_CGAN_denoise(noisy_data=testset_noisy, recover_bar=recover_bar,
                                                        batch_size=128)

            # wt denoise
            # denoise_data = stwt.dwt_soft_threshold(testset_noisy)
            # denoise_data = sp.data_batchlization(denoise_data, batchsize=batch_size)
            # testset_clean = sp.data_batchlization(testset_clean, batchsize=batch_size)

            # idae denoise
            # denoise_data = idae.IDAE_denoise(path_name, testset_noisy)
            # denoise_data = sp.data_batchlization(denoise_data, batchsize=batch_size)
            # testset_clean = sp.data_batchlization(testset_clean, batchsize=batch_size)

            # dnn-cgan denoise
            # denoise_data = dnngan_tls.denoise_DNN_CGAN_denoise(testset_noisy)
            # denoise_data = sp.data_batchlization(denoise_data, batchsize=batch_size)
            # testset_clean = sp.data_batchlization(testset_clean, batchsize=batch_size)

            # post process
            # denoise_data = np.zeros(testset_clean.shape)
            batch_num = denoise_data.shape[0]
            RMSE_piece = np.zeros(batch_num * batch_size)
            for i in range(batch_num):
                for b in range(batch_size):
                    denoise_data[i, b, :, :] = denoise_data[i, b, :, :] + 0.89*recover_bar[i * batch_size + b]
                    testset_clean[i, b, :] = testset_clean[i, b, :] + recover_bar[i * batch_size + b]
                    # RMSE_piece[i * batch_size + b] = metrics.RMSE(denoise_data[i, j, :, :], testset_clean[i, j, :])

            RMSE = metrics.RMSE(denoise_data, testset_clean)
            SNR = metrics.SNR(denoise_data, testset_clean)
            print("RMSE=", RMSE)
            print("SNR=", SNR)
            snr[j, k] = SNR
            rmse[j, k] = RMSE
    snr = np.asarray(snr)
    pd.DataFrame(snr).to_csv(
        (path_name + '/result_snr.csv'))
    rmse = np.asarray(rmse)
    pd.DataFrame(rmse).to_csv(
        (path_name + '/result_rmse.csv'))

# CGAN training
state = False
if state:
    cgan = gan_tls.CGAN()
    cgan.training_ddpm_adapt(train_clean=trainset_clean, train_noisy=trainset_noisy, test_clean=testset_clean,
                             test_noisy=testset_noisy, epochs=1, lr_G=1e-4, lr_D=1e-5, batch_size=128)

# dnn-CGAN training
state = False
if state:
    dnncgan = dnngan_tls.DNNCGAN()
    dnncgan.training_ddpm_adapt(train_clean=trainset_clean, train_noisy=trainset_noisy,
                                epochs=5, lr_G=1e-4, lr_D=1e-4, batch_size=128)

# IDAE training
state = False
if state:
    cgan = idae.IDAE_denoise_train(path_name=path_name,train_clean=trainset_clean,train_noisy=trainset_noisy,
                       batch_size=batch_size,epoch=100)
    # cgan.training_ddpm_adapt(train_clean=trainset_clean, train_noisy=trainset_noisy, test_clean=testset_clean,
    #                          test_noisy=testset_noisy, epochs=1, lr_G=1e-4, lr_D=1e-5, batch_size=128)


