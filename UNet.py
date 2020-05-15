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
    inputs = Input(shape=image_size)
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

def UnetPP(num_class, image_size):
    # TO BE tested
    inputs = Input(shape=image_size)
    #filters_list = [32, 64, 128, 256, 512]
    filters_list = [48, 96, 192, 384, 768]
    # backbone
    conv00 = unet_unit(inputs, filters_list[0], 1)

    conv10 = MaxPooling2D(pool_size=(2, 2))(conv00)
    conv10 = unet_unit(conv10, filters_list[1], 2)

    conv20 = MaxPooling2D(pool_size=(2, 2))(conv10)
    conv20 = unet_unit(conv20, filters_list[2], 3)

    conv30 = MaxPooling2D(pool_size=(2, 2))(conv20)
    conv30 = unet_unit(conv30, filters_list[3], 4)
    conv30 = Dropout(0.5)(conv30)

    conv40 = MaxPooling2D(pool_size=(2, 2))(conv30)
    conv40 = unet_unit(conv40, filters_list[4], 5)
    conv40 = Dropout(0.5)(conv40)
    # nested structure
    conv01 = unet_decoder_unit(conv10, conv00, filters_list[0], 201)
    conv11 = unet_decoder_unit(conv20, conv10, filters_list[1], 211)
    conv21 = unet_decoder_unit(conv30, conv20, filters_list[2], 221)

    skip02 = concatenate([conv00, conv01], axis=3)
    conv02 = unet_decoder_unit(conv11, skip02, filters_list[0], 202)
    skip12 = concatenate([conv10, conv11], axis=3)
    conv12 = unet_decoder_unit(conv21,skip12, filters_list[1], 212)
    skip03 = concatenate([skip02, conv02], axis=3)
    conv03 = unet_decoder_unit(conv12, skip03, filters_list[0], 203)
    # decoder, similar to unet
    conv31 = unet_decoder_unit(conv40, conv30, filters_list[3], 231)
    conv22 = unet_decoder_unit(conv31, concatenate([conv21, conv20], axis=3), filters_list[2], 222)    
    conv13 = unet_decoder_unit(conv22, concatenate([skip12, conv12], axis=3), filters_list[1], 213)
    conv04 = unet_decoder_unit(conv13, concatenate([skip03, conv03], axis=3), filters_list[0], 204)
    
    conv_f2 = Conv2D(filters=2, kernel_size=3, strides=1, 
                   padding='same', activation='relu', 
                   name='conv_f2')(conv04)
    conv_f1 = Conv2D(filters=num_class, kernel_size=1, strides=1, 
                   padding='same', activation='relu', 
                   name='conv_final')(conv_f2)
    return Model(inputs = inputs, outputs = conv_f1)