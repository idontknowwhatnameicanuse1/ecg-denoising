import numpy as np
import matplotlib.pyplot as plt

import DDPM_tools as tls
import denois_RGAN_tools as cgan_tls
import IDAE_tools as IDAE_tls
import soft_threshold_WT_tools as WT_tls

def denoise_single_image_show(clean_data,noisy_data,denoise_data):
    # clean_data.shape=(datadim,channel)
    plt.subplot(221)
    x = range(len(clean_data))
    plt.plot(x, clean_data)
    # plt.ylim([-0.1,1.1])
    plt.title('ground truth')
    plt.subplot(222)
    plt.plot(x, noisy_data)
    # plt.ylim([-0.1,1.1])
    plt.title('noisy signal')
    plt.subplot(223)
    x = range(len(denoise_data))
    plt.plot(x, denoise_data)
    # plt.ylim([-0.1,1.1])
    plt.title('denoise signal')
    plt.subplot(224)
    x = range(len(clean_data))
    plt.plot(x, clean_data, x, denoise_data)
    # plt.ylim([-0.1,1.1])
    plt.title('compare of ground truth and denoise signal')
    plt.tight_layout()
    # muti GPU version
    # plt.savefig(('/home/zhanghaowei/project/denoiseCGAN/train_fig/fig-{}.png'.format(epoch)))
    # plt.savefig(('/root/autodl-tmp/project/train_fig/train_fig-{}.png'.format(epoch)))
    # plt.savefig(('/root/denois_CGAN/train_fig/fig-{}.png'.format(epoch)))
    plt.show()
    plt.close()

def show_img(clean_data,noisy_data,denoise_data,sample_num):
    # clean_data.shape=(batch_num,batch_size,datadim,channel)
    batch_size=clean_data.shape[1]
    # randomly show the clean version
    rand_index=np.arange(batch_size)
    # np.random.shuffle(rand_index)
    rand_index=rand_index[:sample_num]

    # select batch 0 to show
    clean_data=clean_data[0,:,:,:]
    noisy_data=noisy_data[0,:,:,:]
    denoise_data=denoise_data[0,:,:,:]

    for i in range(sample_num):
        rand=rand_index[i]
        clean_data_plot=clean_data[rand,:,:]
        noisy_data_plot=noisy_data[rand,:,:]
        denoise_data_plot=denoise_data[rand,:,:]
        denoise_single_image_show(clean_data_plot, noisy_data_plot, denoise_data_plot)

def loss_curve(loss,opt_num,path_name):
    t=np.arange(opt_num)+1
    plt.plot(t,loss)
    plt.title('loss curve with optimization number')
    plt.savefig((path_name+'/loss_curve/loss_curve.png'))
    plt.show()
    plt.close()

def show_gen_image(batched_set,show_num):
    # batched_set.shape=(batch_num,batch_size,datadim,channel)
    for i in range(show_num):
        rand_batch=batched_set[0,:,:,:]
        gen_sample=rand_batch[i,:,:]
        plt.plot(gen_sample)
        plt.show()
        plt.close()

def demo_plot(path_name,ground_truth,noise_data,denoise_data,start_time,end_time):
    # reshape
    ground_truth=np.reshape(ground_truth,(ground_truth.shape[0]*ground_truth.shape[1]*ground_truth.shape[2],1))
    noise_data_plot=np.reshape(noise_data,(noise_data.shape[0]*noise_data.shape[1]*noise_data.shape[2],1))
    denoise_data=np.reshape(denoise_data,(denoise_data.shape[0]*denoise_data.shape[1]*denoise_data.shape[2],1))

    # reshape plot
    time_s=start_time  # 182500
    time_e=end_time  # 183500
    ground_truth_plot=ground_truth[time_s:time_e]
    noise_data_plot=noise_data_plot[time_s:time_e]
    denoise_data_plot=denoise_data[time_s:time_e]

    print(('full denoise plot'))
    plt.subplot(411)
    plt.plot(ground_truth_plot)
    plt.ylim([0, 1.1])
    plt.title('ground truth')
    plt.subplot(412)
    plt.plot(noise_data_plot)
    plt.ylim([0, 1.1])
    plt.title('noisy signal')
    plt.subplot(413)
    plt.plot(denoise_data_plot)
    plt.ylim([0, 1.1])
    plt.title('denoise signal')
    plt.subplot(414)
    x = range(len(ground_truth_plot))
    plt.plot(x, ground_truth_plot, x, denoise_data_plot)
    plt.ylim([0, 1.1])
    plt.title('compare of ground truth and denoise signal')
    # plt.legend('ground truth','denoise signal')
    plt.tight_layout()
    plt.savefig((path_name+'/denoise_demo.png'))
    plt.show()

def compare_demo_plot(path_name,ground_truth,noise_data,start_time,end_time,
                      beta_t,sigma_t,datadim,batch_size,channel,recover_bar):
    # denoise
    # DDPM denoise
    denoise_data_DDPM= tls.denoise_conditional_sampling(condition=noise_data, factor=0.8, path_name=path_name,
                                                         beta_t=beta_t, sigma_t=sigma_t,
                                                         datadim=datadim, batch_size=batch_size, channel=channel)

    # DNN GAN denoise
    noise_data=np.reshape(noise_data,(noise_data.shape[0],noise_data.shape[1],noise_data.shape[2],1))
    denoise_data_CGAN=cgan_tls.denoise_CGAN_denoise(noise_data, recover_bar,batch_size=128)
    denoise_data_CGAN=np.reshape(denoise_data_CGAN,(denoise_data_CGAN.shape[0],denoise_data_CGAN.shape[1],denoise_data_CGAN.shape[2],1))
    # denoise_data_CGAN= tls.denoise_conditional_sampling(condition=denoise_data_CGAN, factor=0.8, path_name=path_name,
    #                                                      beta_t=beta_t, sigma_t=sigma_t,
    #                                                      datadim=datadim, batch_size=batch_size, channel=channel)


    # WT denoise
    noise_data_for_WT = noise_data[0, :, :, :]
    noise_data_for_WT=noise_data_for_WT[:,:,0]
    noise_data_for_WT=np.reshape(noise_data_for_WT,(noise_data_for_WT.shape[0]*noise_data_for_WT.shape[1]))
    noise_data_for_WT=noise_data_for_WT[:(1034*126)]
    noise_data_for_WT=np.reshape(noise_data_for_WT,(126,1034))
    denoise_data_WT=WT_tls.dwt_soft_threshold(input=noise_data_for_WT)

    # IDAE denoise
    denoise_data_IDAE=IDAE_tls.IDAE_denoise(path_name=path_name,testset_noisy=noise_data_for_WT)


    # recover
    batch_num = denoise_data_DDPM.shape[0]
    for i in range(batch_num):
        for j in range(batch_size):
            denoise_data_DDPM[i, j, :, :] = 1.3*denoise_data_DDPM[i, j, :, :] + recover_bar[i * batch_size + j]
            denoise_data_CGAN[i, j, :] = denoise_data_CGAN[i, j, :] + recover_bar[i * batch_size + j]
            ground_truth[i, j, :] = ground_truth[i, j, :] + recover_bar[i * batch_size + j]
            noise_data[i, j, :] = noise_data[i, j, :]+recover_bar[i * batch_size + j]

    # reshape
    ground_truth=np.reshape(ground_truth,(ground_truth.shape[0]*ground_truth.shape[1]*ground_truth.shape[2],1))
    noise_data_plot=np.reshape(noise_data,(noise_data.shape[0]*noise_data.shape[1]*noise_data.shape[2],1))
    denoise_data_DDPM=np.reshape(denoise_data_DDPM,(denoise_data_DDPM.shape[0]*denoise_data_DDPM.shape[1]*denoise_data_DDPM.shape[2],1))
    denoise_data_CGAN=np.reshape(denoise_data_CGAN,(denoise_data_CGAN.shape[0]*denoise_data_CGAN.shape[1]*denoise_data_CGAN.shape[2],1))
    denoise_data_IDAE=np.reshape(denoise_data_IDAE,(denoise_data_IDAE.shape[0]*denoise_data_IDAE.shape[1],1))
    denoise_data_WT=np.reshape(denoise_data_WT,(denoise_data_WT.shape[0]*denoise_data_WT.shape[1],1))


    # reshape plot
    time_s=start_time  # 182500
    time_e=end_time  # 183500
    ground_truth_plot=ground_truth[time_s:time_e]
    noise_data_plot=noise_data_plot[time_s:time_e]
    denoise_data_DDPM_plot=denoise_data_DDPM[time_s:time_e]
    denoise_data_CGAN_plot=denoise_data_CGAN[time_s:time_e]
    denoise_data_IDAE_plot=denoise_data_IDAE[time_s:time_e]+0.2
    denoise_data_WT_plot=denoise_data_WT[time_s:time_e]+0.3

    print(('full denoise plot'))
    plt.subplot(2,1,1)
    plt.plot(ground_truth_plot)
    plt.ylim([0, 1.1])
    plt.title('ground truth')
    plt.subplot(2,1,2)
    plt.plot(noise_data_plot)
    plt.ylim([0, 1.1])
    plt.title('noisy signal')
    plt.tight_layout()
    plt.savefig((path_name + '/denoise_demo_ground_truth.png'))
    plt.show()
    plt.close()

    plt.subplot(2,1,1)
    plt.plot(denoise_data_WT_plot)
    plt.ylim([0, 1.1])
    plt.title('WT denoise signal')
    plt.subplot(2,1,2)
    x = range(len(ground_truth_plot))
    plt.plot(x, ground_truth_plot, x, denoise_data_WT_plot)
    plt.ylim([0, 1.1])
    plt.title('compare with ground truth(WT)')
    plt.tight_layout()
    plt.savefig((path_name + '/denoise_demo_WT.png'))
    plt.show()
    plt.close()

    plt.subplot(2,1,1)
    plt.plot(denoise_data_IDAE_plot)
    plt.ylim([0, 1.1])
    plt.title('IDAE denoise signal')
    plt.subplot(2,1,2)
    x = range(len(ground_truth_plot))
    plt.plot(x, ground_truth_plot, x, denoise_data_IDAE_plot)
    plt.ylim([0, 1.1])
    plt.title('compare with ground truth(IDAE)')
    plt.tight_layout()
    plt.savefig((path_name + '/denoise_demo_IDAE.png'))
    plt.show()
    plt.close()

    plt.subplot(2,1,1)
    plt.plot(denoise_data_CGAN_plot)
    plt.ylim([0, 1.1])
    plt.title('DNN-CGAN denoise signal')
    plt.subplot(2,1,2)
    x = range(len(ground_truth_plot))
    plt.plot(x, ground_truth_plot)
    plt.plot(x, denoise_data_CGAN_plot)
    plt.ylim([0, 1.1])
    plt.title('compare with ground truth(DNN-CGAN)')
    # plt.legend()
    plt.tight_layout()
    plt.savefig((path_name + '/denoise_demo_DNN-CGAN.png'))
    plt.show()
    plt.close()

    plt.subplot(2,1,1)
    plt.plot(denoise_data_DDPM_plot)
    plt.ylim([0, 1.1])
    plt.title('CNN-CGAN denoise siganl')
    plt.subplot(2,1,2)
    x = range(len(ground_truth_plot))
    plt.plot(x, ground_truth_plot,label='ground truth')
    plt.plot(x, denoise_data_DDPM_plot,label='denoise signal')
    plt.ylim([0, 1.1])
    plt.title('compare with ground truth(CNN-CGAN)')
    # plt.legend()
    plt.tight_layout()
    plt.savefig((path_name + '/denoise_demo_CNN-CGAN.png'))
    plt.show()

    # plt.tight_layout()
    # plt.savefig((path_name+'/denoise_demo.png'))
    # plt.show()

