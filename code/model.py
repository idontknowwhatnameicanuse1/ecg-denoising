import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MaxPooling1D, Conv1DTranspose,Cropping1D,Concatenate,Conv1D,ReLU,UpSampling1D
from tensorflow.keras import Input
import tensorflow as tf

def create_DDPM_model():
    input_layer=keras.Input(shape=(1196,13))  # 1196
    # none*1024*2   1196
    output_1=conv_block(units=64,kernel_size=3)(input_layer)
    # none*1024*64  1192
    input_2=MaxPooling1D(strides=2)(output_1)
    # none*512*64   596
    output_2=conv_block(units=128,kernel_size=3)(input_2)
    # none*512*128  592
    input_3=MaxPooling1D(strides=2)(output_2)
    # none*256*128  296
    output_3=conv_block(units=256,kernel_size=3)(input_3)
    # none*256*256  292
    input_4=MaxPooling1D(strides=2)(output_3)  # pool_size=2
    # none*128*256  146
    output_4=conv_block(units=512,kernel_size=3)(input_4)
    # none*128*512  142
    input_5=MaxPooling1D(strides=2)(output_4)
    # none*64*512   71
    output_5=conv_block(units=1024,kernel_size=3)(input_5)
    # none*64*1024  67
    input_6=Conv1DTranspose(filters=512,strides=2,kernel_size=3)(output_5) # kernel_size=2
    # none*128*512  135
    skip_to_6=self_attention_skip_block(
        q_shape=(output_5.shape[1],output_5.shape[2]),v_shape=(output_4.shape[1],output_4.shape[2])
    )([output_5,output_4])
    input_6 = skip_block_padding_valid(skip_to_tensor=input_6, skip_from_tensor=skip_to_6)
    # none*128*1024 135
    output_6=conv_block(units=512,kernel_size=3)(input_6)
    # none*128*512  131
    input_7=Conv1DTranspose(filters=256,strides=2,kernel_size=3)(output_6) # kernel_size=2
    # none*256*256  263
    skip_to_7=self_attention_skip_block(
        q_shape=(output_6.shape[1], output_6.shape[2]), v_shape=(output_3.shape[1], output_3.shape[2])
    )([output_6,output_3])
    input_7=skip_block_padding_valid(skip_to_tensor=input_7,skip_from_tensor=skip_to_7)
    # none*256*512  263
    output_7=conv_block(units=256,kernel_size=3)(input_7)
    # none*256*256  259
    input_8=Conv1DTranspose(filters=128,strides=2,kernel_size=2)(output_7)
    # none*512*128  518
    skip_to_8=self_attention_skip_block(
        q_shape=(output_7.shape[1], output_7.shape[2]), v_shape=(output_2.shape[1], output_2.shape[2])
    )([output_7,output_2])
    input_8=skip_block_padding_valid(skip_to_tensor=input_8,skip_from_tensor=skip_to_8)
    # none*512*256  518    519
    output_8=conv_block(units=128,kernel_size=3)(input_8)
    # none*512*128  514    515
    input_9=Conv1DTranspose(filters=64,strides=2,kernel_size=2)(output_8)
    # none*1024*64   1028   1030
    skip_to_9=self_attention_skip_block(
        q_shape=(output_8.shape[1], output_8.shape[2]), v_shape=(output_1.shape[1], output_1.shape[2])
    )([output_8,output_1])
    input_9=skip_block_padding_valid(skip_to_tensor=input_9,skip_from_tensor=skip_to_9)
    # none*1024*128  1028   1030
    output_9=conv_block(units=64,kernel_size=3)(input_9)
    # none*1024*64 , output shape is none*(input.shape-172)*64    1026
    output=Conv1D(filters=1,kernel_size=1)(output_9)   # 1024
    # output channel=2, first channel is the predicted gaussian, the second is the vector v for variance opt

    model=keras.Model(inputs=input_layer,outputs=output)
    return model

def conv_block(units,kernel_size):
    model=Sequential()
    model.add(keras.layers.Conv1D(filters=units,
                                  kernel_size=kernel_size,
                                  activation='relu'))
    model.add(keras.layers.Conv1D(filters=units,
                                  kernel_size=kernel_size,
                                  activation='relu'))
    model.compile()
    return model

def skip_block_padding_valid(skip_to_tensor,skip_from_tensor):
    if (skip_from_tensor.shape[1]-skip_to_tensor.shape[1])%2:
        crop = int(tf.cast((skip_from_tensor.shape[1] - skip_to_tensor.shape[1]-1) / 2,dtype=tf.int32))
        crop_left=crop
        crop_right=crop+1
    else:
        crop = int(tf.cast((skip_from_tensor.shape[1] - skip_to_tensor.shape[1]) / 2,dtype=tf.int32))
        crop_left=crop
        crop_right=crop
    output=Cropping1D(cropping=(crop_left,crop_right))(skip_from_tensor)
    output=Concatenate(axis=2)([skip_to_tensor,output])
    return output

def self_attention_skip_block(q_shape,v_shape):
    # attention block
    attention_factor=1
    q_input=Input(shape=q_shape)  # q_input.shape=(none,datadim_q,channel_q)
    v_input=Input(shape=v_shape)  # v_input.shape=(none,datadim_v,channel_v)
    # in normal case 2*datadim_q<datadim_v
    output_channel=v_shape[1]
    # q upsample
    q_upsample=UpSampling1D(size=3)(q_input)
    # q_upsample.shape=(none,2*datadim_q,channel_q)
    # calculate Q and V, and adjust the Q size to the same of V
    V_kernel_size=3
    Q_kernel_size=3*q_shape[0]-v_shape[0]+1
    Q=Conv1D(filters=output_channel,kernel_size=Q_kernel_size)(q_upsample)
    # Q shape=(none,datadim_v,channel=channel_v)
    V=Conv1D(filters=output_channel,kernel_size=V_kernel_size,padding='same')(v_input)
    # V shape=(none,datadim_v,channel=channel_v)
    similarity=ReLU()(Q+V)
    similarity=Conv1D(filters=output_channel,kernel_size=3,padding='same',activation='sigmoid')(similarity)
    # similarity.shape=(none,datadim_v,channel=channel_v)
    similarity=attention_factor*similarity
    output=similarity*v_input
    model=keras.Model(inputs=[q_input,v_input],outputs=output)
    return model

def basic_skip_block(skip_to_tensor,skip_from_tensor):
    output=Concatenate(axis=2)([skip_from_tensor,skip_to_tensor])
    return output
