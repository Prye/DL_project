import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
def unet_unit(input_layer, filter_num, layer_num):
    conv1 = Conv2D(filters=filter_num, kernel_size=3, strides=1, 
                   padding='same', activation='relu', 
                   name='conv'+str(layer_num)+'_1')(input_layer)
    conv2 = Conv2D(filters=filter_num, kernel_size=3, strides=1, 
                   padding='same',  activation='relu', 
                   name='conv'+str(layer_num)+'_2')(input_layer)
    return conv2
    
def unet_decoder_unit(input_layer, skip_layer, filter_num, layer_num):
    upsample = UpSampling2D(size=(2,2))(input_layer)
    conv1 = Conv2D(filters=filter_num, kernel_size=2, strides=1, 
                   padding='same', activation='relu', 
                   name='upconv'+str(layer_num))(upsample)
    skipping = concatenate([skip_layer, conv1], axis=3)
    return unet_unit(skipping, filter_num, layer_num)
    
def Unet(num_class, image_size):
    inputs = Input(shape=[image_size, image_size, 1])
    filters_list = [64, 128, 256, 512, 1024]

    conv1 = unet_unit(inputs, filters_list[0], 1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = unet_unit(pool1, filters_list[1], 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = unet_unit(pool2, filters_list[2], 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = unet_unit(pool3, filters_list[3], 4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = unet_unit(pool4, filters_list[4], 5)
    drop5 = Dropout(0.5)(conv5)

    decode6 = unet_decoder_unit(drop5, drop4, filters_list[3], 6)
    decode7 = unet_decoder_unit(decode6, conv3, filters_list[2], 7)    
    decode8 = unet_decoder_unit(decode7, conv2, filters_list[1], 8)
    decode9 = unet_decoder_unit(decode8, conv1, filters_list[0], 9)

    conv9 = Conv2D(filters=2, kernel_size=3, strides=1, 
                   padding='same', activation='relu', 
                   name='conv9_3')(decode9)
    conv10 = Conv2D(filters=num_class, kernel_size=1, strides=1, 
                   padding='same', activation='relu', 
                   name='conv_final')(conv9)
                   
    return Model(inputs = inputs, outputs = conv10)