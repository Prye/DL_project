import os
import numpy as np
import tensorflow as tf
import skimage
import datetime

import SimpleITK as sitk
from skimage.transform import resize
from skimage.io import imsave
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean,MSE, MeanIoU
from tensorflow.keras.preprocessing.image import apply_affine_transform

from UNet import Unet

optimizer = Adam()

loss_function = BinaryCrossentropy()

train_loss = Mean(name='train_loss')
validation_loss = Mean(name='val_loss')
validation_iou = MeanIoU(num_classes=2, name='val_iou')

image_shape = (224,224,1)
epoch_num = 10*100
batch_size = 10
dataset_fraction = 10

def load_image(image_path):
    # stacking 4 different imaging modality
    all_channel_im_path = os.listdir(image_path)
    im_list = list()
    for di in all_channel_im_path:
        if 'Flair' in di:
            im_dir = os.path.join(image_path, di, di+'.mha')
            img = sitk.ReadImage(im_dir)
            im_list.append(sitk.GetArrayFromImage(img))
    return np.stack(im_list, axis=-1)

def load_mask(image_path):
    # 加载Mask
    all_channel_im_path = os.listdir(image_path)
    for di in all_channel_im_path:
        if 'OT' in di:
            im_dir = os.path.join(image_path, di, di+'.mha')
            img = sitk.ReadImage(im_dir)
            mask = sitk.GetArrayFromImage(img)
            mask = mask.astype(np.bool)  # 有值的地方都是肿瘤
            return mask[:, :, :, np.newaxis]
    raise FileNotFoundError("Label not exist!" + image_path)

def resize_image_batch(batch):
    result = np.empty(shape=(batch.shape[0], image_shape[0], 
                             image_shape[1], batch.shape[3]), dtype=np.int16)
    
    for n in range(batch.shape[0]):
        # 8+224 = 232
        result[n,:,:,:] = batch[n,8:232,8:232,:]
    return result

def get_brats_data(base_path):
    dir_list = os.listdir(base_path)
    data_list = []
    mask_list = []
    # name template => brats_tcia_patxxx_0001
    dir_list = list(filter(lambda x:'brats_tcia_' in x, dir_list))
    for index, item in enumerate(dir_list):
        image_path = os.path.join(base_path, item)
        mri_volume = load_image(image_path)
        mask_volume = load_mask(image_path)
        zero_index= np.sum(mask_volume, axis=1)
        zero_index = np.sum(zero_index, axis=1)
        zero_index = np.sum(zero_index, axis=1)
        data_list.append(resize_image_batch(mri_volume[zero_index>0]))
        mask_list.append(resize_image_batch(mask_volume[zero_index>0]))
    data_list = np.concatenate(data_list)
    mask_list = np.concatenate(mask_list)
    return data_list, mask_list

def data_augmentation(images, labels):
    new_images = []
    new_labels = []
    for idx in range(images.shape[0]*3):
        shifts = np.random.randint(-10,10,2)
        rotation = np.random.randint(-20,20)
        zoom = 1.15 - np.random.random(2)/4
        new_im = apply_affine_transform(
            images[idx//3], theta=rotation, tx=shifts[0], ty=shifts[1],
            zx=zoom[0], zy=zoom[1], fill_mode='constant', cval=0.0
        )
        new_label = apply_affine_transform(
            labels[idx//3], theta=rotation, tx=shifts[0], ty=shifts[1],
            zx=zoom[0], zy=zoom[1], fill_mode='constant', cval=0.0
        )
        new_images.append(new_im)
        new_labels.append(new_label)
    new_images = np.stack(new_images)
    new_labels = np.stack(new_labels)
    images = np.concatenate([images,new_images])
    labels = np.concatenate([labels, new_labels])
    return images, labels

def prepare_brats15_npy():
    # SimpleITK is slow, preprocess data to save time
    base_data_path = os.path.join("BRATS2015_Training",'HGG')
    all_data, all_label = get_brats_data(base_data_path)
    data_order = np.arange(all_data.shape[0])
    np.random.shuffle(data_order)
    all_data = all_data[data_order,:,:,:]/ 4095  # MRI pixel range
    all_label = all_label[data_order,:,:,:].astype('float64')
    pack_size = all_data.shape[0] // dataset_fraction
    print('all data shape:', all_data.shape)
    print('all label shape:', all_label.shape)
    for i in range(dataset_fraction):
        np.save('training_data_%d.npy'%i,all_data[i*pack_size:(i+1)*pack_size])
        np.save('training_labe_%d.npy'%i, all_label[i*pack_size:(i+1)*pack_size])
    
#prepare_brats15_npy()

def train_brats15_subsets():
    # 数据准备
    #  分配训练，验证，测试
    validation_data = np.load('training_data_0.npy')
    validation_label = np.load('training_labe_0.npy')
    validation_order = np.arange(validation_data.shape[0])
    #train_data, train_label = data_augmentation(train_data, train_label)
    model = Unet(1, image_shape)
    step_counter = 0

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/UNet_Brats_'+ current_time
    model_path = os.path.join(train_log_dir, 'best_model.h5')
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    best_iou = -1
    step_in_epoch = (validation_data.shape[0] // batch_size) + 1
    # 训练环节
    for ep in range(epoch_num):
        train_data = np.load('training_data_%d.npy'%(2+(ep%(dataset_fraction-2))))
        train_label = np.load('training_labe_%d.npy'%(2+(ep%(dataset_fraction-2))))
        for step in range(step_in_epoch):
            step_counter += 1
            batch_start = step*batch_size
            batch_end = batch_start + batch_size
            batch_data = train_data[batch_start:batch_end, :,:,:]
            batch_label = train_label[batch_start:batch_end, :,:,:]
            with tf.GradientTape() as tape:
                predictions = model(batch_data, training=True)
                loss = loss_function(batch_label, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            with summary_writer.as_default():
                tf.summary.scalar('Stats/train_loss', 
                                  train_loss.result(), step=step_counter)
            train_loss.reset_states()
        # validation step
        validation_loss_list = []
        validation_iou_list = []
        for i in range(20):
            val_batch_label = validation_label[i*batch_size:(i+1)*batch_size]
            predictions = model.predict(validation_data[i*batch_size:(i+1)*batch_size])
            v_loss = loss_function(val_batch_label, predictions)
            validation_loss(v_loss)
            predictions[predictions>=0.5] = 1.0
            predictions[predictions<0.5] = 0.0
            validation_iou.update_state(val_batch_label, predictions)
            validation_iou_list.append(validation_iou.result())
            validation_loss_list.append(validation_loss.result())
            validation_loss.reset_states()
            validation_iou.reset_states()
        
        validation_data = validation_data[validation_order]
        validation_label = validation_label[validation_order]
        np.random.shuffle(validation_order)
        validation_score_loss = np.mean(validation_loss_list)
        validation_score_iou = np.mean(validation_iou_list)
        with summary_writer.as_default():
            tf.summary.scalar('stats/val_loss', validation_score_loss, step=ep)
            tf.summary.scalar('stats/val_iou', validation_score_iou, step=ep)
        if validation_score_iou > best_iou:
            model.save_weights(model_path)
            best_iou = validation_score_iou

    print("best validation iou:", best_iou)
    
    
    # test step
    test_data = np.load('training_data_1.npy')
    test_label = np.load('training_labe_1.npy')
    model.load_weights(model_path)
    test_loss_list = []
    test_iou_list = []
    for step in range(step_in_epoch):
        batch_start = step*batch_size
        batch_end = batch_start + batch_size
        batch_data = test_data[batch_start:batch_end]
        batch_label = test_label[batch_start:batch_end]
        test_pred = model.predict(batch_data)
        t_loss = loss_function(batch_label, test_pred)
        validation_loss(t_loss)
        test_loss_list.append(validation_loss.result())
        validation_loss.reset_states()
        test_pred[test_pred>=0.5] = 1.0
        test_pred[test_pred<0.5] = 0.0
        
        # 打印test set里面的图片 10 zhang
        if step==0:
            for im_idx in range(test_pred.shape[0]):
                imsave(os.path.join(train_log_dir, 'test_%d.png'%im_idx), test_pred[im_idx])
                imsave(os.path.join(train_log_dir, 'test_label_%d.png'%im_idx), batch_label[im_idx])
    
        validation_iou.update_state(batch_label, test_pred)
        test_iou_list.append(validation_iou.result())
        validation_iou.reset_states()
    print('test mIoU:', np.mean(test_iou_list))
    print('test loss:', np.mean(test_loss_list))

train_brats15_subsets()


