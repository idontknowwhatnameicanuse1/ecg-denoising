import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MaxPooling1D, Conv1DTranspose,Cropping1D,Concatenate,Conv1D,ReLU,UpSampling1D
from tensorflow.keras import Input
import DDPM_tools as tls
import numpy as np
import pre_process as sp
from model_experiment import create_DDPM_model

def encoder_model():
    c=1
    model=Sequential()
    model.add(keras.Input(shape=(1024,1)))
    # model.input.shape=(none,1024,1)
    model.add(Conv1D(filters=c*64,kernel_size=5,padding='same',activation='relu'))
    # shape=(none,1024,64)
    model.add(MaxPooling1D(pool_size=2))
    # shape=(none,512,64)
    model.add(Conv1D(filters=c*128,kernel_size=5,padding='same',activation='relu'))
    # shape=(none,512,128)
    model.add(MaxPooling1D(pool_size=2))
    # shape=(none,256,128)
    model.add(Conv1D(filters=c*256,kernel_size=5,padding='same',activation='relu'))
    # shape=(none,256,256)
    model.add(MaxPooling1D(pool_size=2))
    # shape=(none,128,256)
    model.add(Conv1D(filters=c*512, kernel_size=5, padding='same',activation='relu'))
    # shape=(none,128,512)
    model.add(MaxPooling1D(pool_size=2))
    # shape=(none,64,512)
    model.add(Conv1D(filters=c*1024, kernel_size=5, padding='same',activation='relu'))
    # shape=(none,64,1024)
    model.add(MaxPooling1D(pool_size=2))
    # shape=(none,32,1024)

    # through
    model.add(Conv1D(filters=c*1024, kernel_size=5, padding='same',activation='relu'))
    # shape=(none,32,1024)
    model.add(UpSampling1D(size=2))
    # shape=(none,64,1024)
    model.add(Conv1D(filters=c*512, kernel_size=5, padding='same',activation='relu'))
    # shape=(none,64,512)
    model.add(UpSampling1D(size=2))
    # shape=(none,128,512)
    model.add(Conv1D(filters=c*256, kernel_size=5, padding='same',activation='relu'))
    # shape=(none,128,256)
    model.add(UpSampling1D(size=2))
    # shape=(none,256,256)
    model.add(Conv1D(filters=c*128, kernel_size=5, padding='same',activation='relu'))
    # shape=(none,256,128)
    model.add(UpSampling1D(size=2))
    # shape=(none,512,128)
    model.add(Conv1D(filters=c*64, kernel_size=5, padding='same',activation='relu'))
    # shape=(none,512,64)
    model.add(UpSampling1D(size=2))
    # shape=(none,1024,64)

    model.add(Conv1D(filters=1, kernel_size=1))
    # shape=(none,1024,1)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


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


def encoder_model_train(clean_data,noisy_data,beta_t,batch_size,path_name):
    T=beta_t.shape[0]
    batch_num=clean_data.shape[0]

    # clean_forward_encoder=np.zeros(shape=(batch_num*batch_size,clean_data.shape[2],clean_data.shape[3]))
    # noisy_forward_encoder=np.zeros(shape=(batch_num*batch_size,noisy_data.shape[2],noisy_data.shape[3]))
    #
    # # clean forward
    # for i in range(batch_num):
    #     print('batch num = ', i)
    #     rand_timestep = np.random.randint(0, 0.1*T, size=(batch_size, 1))
    #     # rand_timestep=np.reshape(rand_timestep,(batch_size,1)), range [0 T)
    #     rand_timestep = (rand_timestep + 1).astype(int)
    #
    #     # noisy_sample=np.reshape(noisy_data[i,:,:,0],(noisy_data.shape[1],noisy_data.shape[2],1))
    #     noisy_sample=noisy_data[i,:,:,:]
    #
    #     clean_forward_encoder_batched, noisy_forward_encoder_batched = tls.DDPM_forward_direct_sample_for_denoise(
    #         clean_sample=clean_data[i,:,:,:], noisy_sample=noisy_sample,
    #         beta_t=beta_t, t_batch=rand_timestep, dif_mod=1,
    #         label_gaussian=0)
    #
    #     # diffusion step encoding
    #     noisy_forward_encoder_batched_time_encoded = sp.net_input_phase_encoding(data=noisy_forward_encoder_batched, T=T, t=rand_timestep)
    #
    #     if i>0:
    #         clean_forward_encoder=np.concatenate([clean_forward_encoder,clean_forward_encoder_batched],axis=0)
    #         noisy_forward_encoder=np.concatenate([noisy_forward_encoder,noisy_forward_encoder_batched_time_encoded],axis=0)
    #     else:
    #         clean_forward_encoder=clean_forward_encoder_batched
    #         noisy_forward_encoder=noisy_forward_encoder_batched_time_encoded

    model=encoder_model()
    # model=create_generator_12lead()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    print('model training')
    # model.fit(x=noisy_forward_encoder,y=clean_forward_encoder,batch_size=batch_size,epochs=50, verbose=1)
    model.fit(x=noisy_data,y=clean_data,batch_size=batch_size,epochs=10, verbose=1)
    model.save(path_name+'/saved_model/encoder_model')


