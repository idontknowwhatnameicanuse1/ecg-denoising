import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Model
# from unpadding_unet import create_DDPM_model
from model import create_DDPM_model
from scipy import signal
import pre_process as sp
import plot_tools as ptls
import metrics as ms

def file_path(servier):
    if servier=='autodl':
        path_name='/root/autodl-tmp/ddpm'
    if servier=='hust_individual':
        path_name='/root/DDPM'
    if servier=='hust_cluster':
        path_name='/home/uu201912628/DDPM'
    if servier=='hust_exp':
        path_name='/home/zhanghaowei/DDPM'
    return path_name

def cosine_schedule(T,s):
    t=np.arange(T)
    beta_t=1-(np.cos(np.pi/2*(t/T+s)/(1+s)))**2/((np.cos(np.pi/2*((t-1)/T+s)/(1+s)))**2)
    beta_t[0]=0.0001
    beta_t[beta_t>0.999]=0.999
    return beta_t

def DDPM_forward_direct_sample(clean_sample,beta_t,t_batch,dif_mod,label_gaussian):
    # clean_sample is 1-dimensional signal, shape=(batch_size, datadim, channel=1)
    # beta is a array, beta.shape=(T,1), T is the total step
    # direct sample of xt, under the condition of small beta_t
    # t is a vector, each sample in the batch has its own t
    # t can only be larger than 1, t range 1:T

    # parameter reading
    batch_size=clean_sample.shape[0]
    datadim=clean_sample.shape[1]
    channel=clean_sample.shape[2]
    T=beta_t.shape[0]
    x_t=np.zeros((batch_size,datadim,channel))

    for i in range(batch_size):
        t=(t_batch[i]-1).astype(int)
        x0=clean_sample[i,:,:]
        # ensure t is under the boundary of T
        # if t > T:
        #     print("out of the iter time T")
        #     t = T
        # if t < 1:
        #     t = 1

        # beta and alpha from 0 to t
        # beta_to_t = beta_t[:t, :]
        alpha = 1 - beta_t
        beta_t_hat = np.cumprod(alpha)
        # beta_t_hat = np.reshape(beta_t_hat, (beta_t_hat.shape[0], 1))
        beta_t_hat = beta_t_hat[t]

        if dif_mod:  # to test the gaussian should to same to the label gaussian
            # sample of gaussian
            z_t = np.random.normal(size=(datadim, channel))
        else:
            z_t=label_gaussian

        # forward diffussion
        x_t[i,:,:] = np.sqrt(beta_t_hat) * x0 + z_t * np.sqrt(1 - beta_t_hat)

    return x_t

def DDPM_forward_direct_sample_for_denoise(clean_sample,noisy_sample,beta_t,t_batch,dif_mod,label_gaussian):
    # clean_sample is 1-dimensional signal, shape=(batch_size, datadim, channel=1)
    # beta is a array, beta.shape=(T,1), T is the total step
    # direct sample of xt, under the condition of small beta_t
    # t is a vector, each sample in the batch has its own t
    # t can only be larger than 1, t range 1:T

    # parameter reading
    batch_size=clean_sample.shape[0]
    datadim=clean_sample.shape[1]
    channel=clean_sample.shape[2]
    T=beta_t.shape[0]
    x_t_clean=np.zeros((batch_size,datadim,channel))
    x_t_noisy=np.zeros((batch_size,datadim,channel))

    for i in range(batch_size):
        t=(t_batch[i]-1).astype(int)
        x0_clean=clean_sample[i,:,:]
        x0_noisy=noisy_sample[i,:,:]

        # beta and alpha from 0 to t
        # beta_to_t = beta_t[:t, :]
        alpha = 1 - beta_t
        beta_t_hat = np.cumprod(alpha)
        # beta_t_hat = np.reshape(beta_t_hat, (beta_t_hat.shape[0], 1))
        beta_t_hat = beta_t_hat[t]

        if dif_mod:  # to test the gaussian should to same to the label gaussian
            # sample of gaussian
            z_t = np.random.normal(size=(datadim, channel))
        else:
            z_t=label_gaussian

        # forward diffussion
        x_t_clean[i,:,:] = np.sqrt(beta_t_hat) * x0_clean + z_t * np.sqrt(1 - beta_t_hat)
        x_t_noisy[i,:,:] = np.sqrt(beta_t_hat) * x0_noisy + z_t * np.sqrt(1 - beta_t_hat)

    return x_t_clean,x_t_noisy


class DDPM_model(Model):
    def __init__(self):
        super(DDPM_model, self).__init__()
        self.layer = create_DDPM_model()
        self.layer.compile()
        self.layer.summary()

    def Lvlb(self, clean_data, beta_t):
        T = beta_t.shape[0]
        batch_size = 10# clean_data.shape[0]
        clean_data=clean_data[:batch_size,:,:]
        Lt_sum = 0
        for i in range(1, T - 1):
            t = ((i + 1) * np.ones((batch_size, 1))).astype(int)  # t range [1 T]
            t_index = (t[0]).astype(int)
            x_t_batch = DDPM_forward_direct_sample(clean_sample=clean_data,
                                                   beta_t=beta_t, t_batch=t, dif_mod=0, label_gaussian=0)
            # x_t_batch.shape=(batch_size,datadim,channel)
            # concat the timestep message with forward sample
            x_t_batch = sp.net_input_phase_encoding(data=x_t_batch, T=T, t=t)

            # model output has two channel, the first is predicted gaussian, the second is vector v
            model_output=self.layer(x_t_batch)
            epsinon = model_output[:, :, 0]
            epsinon = np.reshape(epsinon, (epsinon.shape[0], epsinon.shape[1], 1))
            v = model_output[:, :, 1]
            v = np.reshape(v, (v.shape[0], v.shape[1], 1))
            v = np.mean(v)

            # count for alpha and beta
            alpha = 1 - beta_t
            beta_t_hat = np.cumprod(alpha)
            # beta_t_hat = np.reshape(beta_t_hat, (beta_t_hat.shape[0], 1))
            alpha_t_hat = beta_t_hat[t_index - 1]
            alpha_t_1_hat = beta_t_hat[(t_index - 2).astype(int)]
            alpha_t = alpha[t_index]
            beta_t_use = beta_t[t_index]
            beta_t_wave = (1 - alpha_t_1_hat) / (1 - alpha_t_hat) * beta_t_use

            # miu and sigma of two gaussian
            miu_q = beta_t_use * np.sqrt(alpha_t_1_hat) / (1 - alpha_t_hat) * clean_data + np.sqrt(alpha_t) * (
                        1 - alpha_t_1_hat) / (1 - alpha_t_hat) * x_t_batch
            sigma_q = beta_t_wave
            miu_p = 1 / np.sqrt(alpha_t) * (
                        x_t_batch - (beta_t_use) / (1 - alpha_t_hat) * epsinon)  # shape=(batch_size,datadim,channel)
            sigma_p = np.exp(v * np.log(beta_t_use) + (1 - v) * np.log(beta_t_wave))
            Lt = KLdiv_gaussian(miu_q, miu_p, sigma_q, sigma_p)  # Lt.shape=(batch_size,1)
            factor = 0.2
            Lt_sum = Lt_sum + factor * Lt / np.sqrt(np.mean(Lt ** 2))
        loss = np.mean(Lt_sum)
        return loss

    def loss(self, model_input, label, clean_data, beta_t):
        model_output=self.layer(model_input)
        epsinon=model_output[:,:,0]
        label_ex=np.zeros((clean_data.shape[0],clean_data.shape[1]))
        for i in range(clean_data.shape[0]):
            label_ex[i,:]=label[:,0]
        label=label_ex
        # Lvlb count
        # loss_vlb=self.Lvlb(clean_data,beta_t)
        loss_simple=tf.reduce_mean((label-epsinon)**2)
        lamuda=1
        loss_value=loss_simple# +lamuda*loss_vlb
        # loss_value=loss_vlb
        # print("loss=",loss_value,"\n ,vlb=",loss_vlb," ,simple=",loss_simple)
        print("loss=",loss_value)
        return loss_value

    def get_grad(self, model_input, label, clean_data,beta_t):
        with tf.GradientTape() as tape:
            tape.watch(self.layer.variables)
            L = self.loss(model_input,label,clean_data,beta_t)
            g = tape.gradient(L, self.layer.variables)
        return g, L

    def train(self, trainset, epochs, beta_t, path_name,lr=1e-4):
        print("training")
        # batched trainset.shape=(batch_num, batch_size, datadim, 1)
        # parameter reading
        batch_num=trainset.shape[0]
        batch_size=trainset.shape[1]
        datadim = trainset.shape[2]
        channel=trainset.shape[3]
        T=beta_t.shape[0]
        loss=np.zeros(1)

        # trainset_num=trainset.shape[0]
        for e in range(epochs):
            print("epoch:",e)
            # clean data in the clean data distribution
            for _ in tqdm(range(batch_num)):
                train_batch=trainset[_,:,:,:]

                # random select a time step, a batch with same timestep
                rand_timestep = np.random.randint(0, T, size=(batch_size,1))
                # rand_timestep=np.reshape(rand_timestep,(batch_size,1)), range [0 T)
                rand_timestep=(rand_timestep+1).astype(int)

                # gaussian noise as the label
                gaussian = np.random.normal(size=(datadim, 1))
                label = gaussian

                # forward sample of given time step
                x_t = DDPM_forward_direct_sample(train_batch, beta_t, rand_timestep,
                                                 dif_mod=0,label_gaussian=label)
                # x_t.shape=(batch_size,datadim,channel)
                # plt.plot(x_t[0,:,:])
                # plt.show()

                # padding tile before input the model
                x_t_pad=sp.padding_tile(padding_data=x_t,padding_size=172/2)

                # concat the timestep message with forward sample
                x_t_ = sp.net_input_phase_encoding(data=x_t_pad,T=T,t=rand_timestep)

                # optimization
                grad, loss_ = self.get_grad(x_t_, label,clean_data=train_batch,beta_t=beta_t)
                loss_=np.reshape(loss_,(1))
                if (_==0)&(e==0):
                    loss[0]=loss_
                else:
                    loss=np.concatenate([loss,loss_])
                tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(zip(grad, self.layer.variables))
            ptls.loss_curve(loss=loss,opt_num=(e+1)*batch_num,path_name=path_name)

        # save the model
        model=self.layer
        model.save(path_name+'/saved_model/DDPM_model')

    def sampling(self,beta_t,sigma_t,datadim,batch_size,channel):
        print("sampling")
        T=beta_t.shape[0]

        # final version of diffussion of gaussian
        x_T=np.random.normal(size=(batch_size,datadim,channel))

        # final version as the initial state of iter
        x_t=x_T

        # from back to begin of iter
        for i in range(T):
            t=(T-i-1)*np.ones((batch_size,1))
            if t[0] > 1:
                z = np.random.normal(size=(batch_size, datadim, channel))
            else:
                z = 0

            # beta and alpha of step t
            t_index=t[0].astype(np.int)
            print(("step t= "+str(t_index)))
            alpha = 1 - beta_t
            beta_t_hat = np.cumprod(alpha)
            alpha_t = alpha[t_index]
            beta_t_hat_t = beta_t_hat[t_index]

            # padding tile before input the model
            x_t_pad = sp.padding_tile(padding_data=x_t, padding_size=172 / 2)

            # concat the time step message with sample in time step t
            x_t_ = sp.net_input_phase_encoding(data=x_t_pad, T=T, t=t)

            # sampling of the next step t-1
            miu = 1 / np.sqrt(alpha_t) * (x_t - ((1 - alpha_t) / np.sqrt(1 - beta_t_hat_t) * self.layer(x_t_)))
            x_t = miu + sigma_t[t_index] * z
            # x_t.shape=(batch_size,datadim,channel)

            # check the sample of every step
            # plt.plot(x_t[0,:,:])
            # plt.title(('step='+str(t_index)))
            # plt.show()

        # end of the sampling
        x0=x_t

        # show the clean version
        plt.plot(x0[0,:,:])
        plt.show()

def denoise_conditional_sampling(condition,ground,path_name,beta_t,sigma_t,datadim,batch_size,channel):
    # condition shape=(batch_num,batch_size,datadim,channel)
    print("reload conditional sampling")
    model=tf.keras.models.load_model(path_name+'/saved_model/DDPM_model')
    model.compile()
    encoder=tf.keras.models.load_model(path_name+'/saved_model/encoder_model')

    T = beta_t.shape[0]
    batch_num=condition.shape[0]
    denoise_data=np.zeros((batch_num,batch_size,datadim,channel))
    snr_curve = np.zeros(T)
    rmse_curve = np.zeros(T)
    plot_sample_curve = True
    # batch_size=sample_num
    print('total test batch num = ', batch_num)
    for j in range(batch_num):
        print('test batch = ', j)
        # condition batch
        condition_batch=condition[j,:,:,:]

        # final version of diffussion of gaussian
        x_T = np.random.normal(size=(batch_size, datadim, channel))

        # final version as the initial state of iter
        x_t = x_T

        if j == 8:
            a = 1

        # from back to begin of iter
        for i in range(T):  # i range [0 T)
            t = (T - i-1) * np.ones((batch_size, 1))  # t range [1 T]
            t = t.astype(int)
            print('step=', t[0])
            if t[0] > 1:
                z = np.random.normal(size=(batch_size, datadim, channel))
            else:
                z = 0

            if t[0] == 20:
                a=1

            # beta and alpha of step t
            t_index = (t[0]).astype(int)   # t_index range [0 T-1]
            # print(("step t= " + str(t_index)))
            alpha = 1 - beta_t
            beta_t_hat = np.cumprod(alpha)
            alpha_t = alpha[t_index]
            beta_t_hat_t = beta_t_hat[t_index]

            # padding tile before input the model
            x_t_pad = sp.padding_tile(padding_data=x_t, padding_size=172 / 2)   # 174

            # concat the time step message with sample in time step t
            x_t_ = sp.net_input_phase_encoding(data=x_t_pad, T=T, t=t)

            # sampling of the next step t-1
            model_output=model.predict(x_t_)
            epsinon=model_output# [:,:,0]
            # epsinon=np.reshape(epsinon,(epsinon.shape[0],epsinon.shape[1],1))

            miu = 1 / np.sqrt(alpha_t) * (x_t - ((1 - alpha_t) / np.sqrt(1 - beta_t_hat_t) * epsinon))

            # unconditional sample
            x_t = miu + sigma_t[t_index] * z
            # x_t.shape=(batch_size,datadim,channel)

            # time bar encoding
            # noisy_condition=condition_batch[:,:,0]
            # noisy_condition=np.reshape(noisy_condition,(noisy_condition.shape[0],noisy_condition.shape[1],1))
            noisy_condition = condition_batch

            # conditional operation
            # condition is the conditional signal with shape=(batch_size,datadim,channel)
            # to ensure the freq band of final sample is the same with conditional signal,
            # use band pass filter on conditional signal and replace the unconditional
            # sample's band freq with conditional band freq
            # forward of conditional signal, conditional encoding
            # forward_condition = DDPM_forward_direct_sample(condition_batch, beta_t=beta_t, t_batch=t, dif_mod=1,
            #                                            label_gaussian=0)
            forward_condition = DDPM_forward_direct_sample(noisy_condition, beta_t=beta_t, t_batch=t, dif_mod=1,
                                                       label_gaussian=0)

            # encoder model encode the noisy forward condition
            # forward_condition_time_encoded = sp.net_input_phase_encoding(data=forward_condition, T=T, t=t)
            # encoded_condition=encoder.predict(forward_condition_time_encoded)
            encoded_condition=encoder.predict(forward_condition)

            # band replace
            factor_t= 0.8# ((t_index/T-0)*(t_index/T-0>=0))**(1/16) # 1-0.006*t_index # factor[t_index]
            if t[0]>1:
                x_t = (1-factor_t)*x_t+factor_t*encoded_condition
                # x_t = x_t - fai(x_t, batch_size, datadim, channel) + fai(forward_condition, batch_size, datadim, channel)

            if plot_sample_curve:
                snr_curve[i] = ms.SNR(input=x_t, ground_truth=ground)
                rmse_curve[i] = ms.RMSE(input=x_t, ground_truth=ground)
                print('snr = ', snr_curve[i])
                print('rmse = ', rmse_curve[i])
        # plot_sample_curve = False

            if j==0 and (i==0 or i==20 or i==40 or i==60 or i==80 or i>85):
                # check the sample of every step
                plt.plot(x_t[0,:,1])
                plt.title(('step='+str(t_index)))
                plt.show()
            if j==0:
                plt.plot(x_t[0, :, 1])
                plt.title(('step=' + str(t_index)))
                # plt.savefig((path_name + '/saved_fig/step' + str(i) + '.png'))
                plt.show()
                plt.close()
            if j==0:
                plt.plot(ground[0, 0, :, 1])
                plt.title('ground')
                # plt.savefig((path_name + '/saved_fig/step' + str(i) + '.png'))
                plt.show()
                plt.close()

        # end of the sampling
        denoise_data[j,:,:,:] = x_t
    if plot_sample_curve:
        # plt.plot(snr_curve)
        # plt.show()
        # plt.close()
        plt.plot(rmse_curve)
        plt.show()
    return denoise_data

def model_reload_conditional_sampling(condition,factor,path_name,beta_t,sigma_t,datadim,batch_size,channel):
    # condition shape=(batch_num,batch_size,datadim,channel)
    print("reload conditional sampling")
    model=tf.keras.models.load_model(path_name+'/saved_model/DDPM_model')
    model.compile()
    T = beta_t.shape[0]
    batch_num=condition.shape[0]
    denoise_data=np.zeros((batch_num,batch_size,datadim,channel))
    # batch_size=sample_num
    for j in range(batch_num):
        # condition batch
        condition_batch=condition[j,:,:,:]

        # final version of diffussion of gaussian
        x_T = np.random.normal(size=(batch_size, datadim, channel))

        # final version as the initial state of iter
        x_t = x_T

        # from back to begin of iter
        for i in range(T):  # i range [0 T)
            t = (T - i-1) * np.ones((batch_size, 1))  # t range [1 T]
            t=t.astype(int)
            if t[0] > 1:
                z = np.random.normal(size=(batch_size, datadim, channel))
            else:
                z = 0

            # beta and alpha of step t
            t_index = (t[0]).astype(int)   # t_index range [0 T-1]
            print(("step t= " + str(t_index)))
            alpha = 1 - beta_t
            beta_t_hat = np.cumprod(alpha)
            alpha_t = alpha[t_index]
            beta_t_hat_t = beta_t_hat[t_index]

            # padding tile before input the model
            x_t_pad = sp.padding_tile(padding_data=x_t, padding_size=174 / 2)

            # concat the time step message with sample in time step t
            x_t_ = sp.net_input_phase_encoding(data=x_t_pad, T=T, t=t)

            # sampling of the next step t-1
            model_output=model.predict(x_t_)
            epsinon=model_output# [:,:,0]
            # epsinon=np.reshape(epsinon,(epsinon.shape[0],epsinon.shape[1],1))

            miu = 1 / np.sqrt(alpha_t) * (x_t - ((1 - alpha_t) / np.sqrt(1 - beta_t_hat_t) * epsinon))

            # unconditional sample
            x_t = miu + sigma_t[t_index] * z
            # x_t.shape=(batch_size,datadim,channel)

            # conditional operation
            # condition is the conditional signal with shape=(batch_size,datadim,channel)
            # to ensure the freq band of final sample is the same with conditional signal,
            # use band pass filter on conditional signal and replace the unconditional
            # sample's band freq with conditional band freq
            # forward of conditional signal, conditional encoding
            forward_condition = DDPM_forward_direct_sample(condition_batch, beta_t=beta_t, t_batch=t, dif_mod=1,
                                                       label_gaussian=0)
            # band replace
            factor_t=0.5 # factor[t_index]
            if t[0]>1:
                x_t = (1-factor_t)*x_t+factor_t*forward_condition
                # x_t = x_t - fai(x_t, batch_size, datadim, channel) + fai(forward_condition, batch_size, datadim, channel)

            if j==0 and (i==0 or i==20 or i==40 or i==60 or i==80 or i>85):
                # check the sample of every step
                plt.plot(x_t[0,:,:])
                plt.title(('step='+str(t_index)))
                plt.show()
            # if j==0:
            #     plt.plot(x_t[0, :, :])
            #     plt.title(('step=' + str(t_index)))
            #     plt.savefig((path_name + '/saved_fig/step' + str(i) + '.png'))
            #     plt.show()
            #     plt.close()

        # end of the sampling
        denoise_data[j,:,:,:] = x_t
    return denoise_data

def fai(input,batch_size,datadim,channel):
    # fai is a function to extract the band freq of given signal
    # signal.shape=(batch_size,datadim,channel)
    fs=360
    order=4
    cfl=6 # cutoff freq low
    cfh=45 # cutoff freq high
    wn1=2*cfl/fs
    wn2=2*cfh/fs
    wn=[wn1,wn2]
    s=np.zeros((batch_size,datadim,channel))
    # band pass
    b, a=signal.butter(order,wn,'bandpass',output='ba')
    for i in range(batch_size):
        s[i,:,:]=signal.filtfilt(b,a,input[i,:,:],axis=0)
    return s

def KLdiv_gaussian(miu_p,miu_q,sigma_p,sigma_q):
    # calculate the KL divergency between two gaussian distribution
    # miu_p and miu_q has a shape of (batch_size,datadim,1), and sigma_p and sigma_q is a number
    n=1# miu_p.shape[0]
    Dkl_pq=1/2*(np.log(sigma_q/sigma_p)-n
                +sigma_p/sigma_q+1/sigma_q*np.sum((miu_p-miu_q)**2,axis=1))
    # Dkl_pq=Dkl_pq[:,0,:]
    return Dkl_pq


