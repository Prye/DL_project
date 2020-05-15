import os
import numpy as np
import tensorflow as tf
import skimage
import datetime

from libtiff import TIFF, TIFFimage
from skimage.transform import resize
from skimage.io import imsave
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean,MSE, MeanIoU
from tensorflow.keras.preprocessing.image import apply_affine_transform

from UNet import UnetPP

optimizer = Adam()

loss_function = BinaryCrossentropy()

train_loss = Mean(name='train_loss')
validation_loss = Mean(name='val_loss')
validation_iou = MeanIoU(num_classes=2, name='val_iou')

image_shape = ((224,224,1))
epoch_num = 0
batch_size = 8

def openTIF(path):
    tif = TIFF.open(path)
    image_list = []
    for im in tif.iter_images(): 
        image_list.append(resize(im, image_shape))
    return np.stack(image_list)
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

        
def train_EM_Seg():
    # 数据准备
    train_data_path = os.path.join("EMseg",'train-volume.tif')
    train_label_path = os.path.join("EMseg",'train-labels.tif')
    test_data_path = os.path.join("EMseg",'test-volume.tif')

    train_data = openTIF(train_data_path)
    train_label = openTIF(train_label_path)
    test_data = openTIF(test_data_path)
    train_label[train_label>0.5] = 1.0
    train_label[train_label<0.5] = 0.0
    
    validation_data = train_data[:3, :, :, :]
    validation_label = train_label[:3, :, :, :]
    train_data = train_data[3:, :, :, :]
    train_label = train_label[3:, :, :, :]
    train_data, train_label = data_augmentation(train_data, train_label)
    model = UnetPP(1, image_shape)
    step_counter = 0
    data_order = np.arange(train_data.shape[0])
    step_in_epoch = (train_data.shape[0] // batch_size) + 1
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/UnetPP_EM_SEG_'+ current_time
    model_path = os.path.join(train_log_dir, 'best_model.h5')
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    best_iou = -1
    print('train dataset shape:', train_data.shape)
    print('train label shape:', train_label.shape)
    # 训练环节
    for ep in range(epoch_num):
        # shuffle
        np.random.shuffle(data_order)
        train_data = train_data[data_order,:,:,:]
        train_label = train_label[data_order,:,:,:]
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
        
        predictions = model.predict(validation_data)
        v_loss = loss_function(validation_label, predictions)
        validation_loss(v_loss)
        predictions[predictions>=0.5] = 1.0
        predictions[predictions<0.5] = 0.0
        validation_iou.update_state(validation_label, predictions)

        with summary_writer.as_default():
            tf.summary.scalar('stats/val_loss', validation_loss.result(), step=ep)
            tf.summary.scalar('stats/val_iou', validation_iou.result(), step=ep)
        if validation_iou.result() > best_iou:
            model.save_weights(model_path)
            best_iou = validation_iou.result()
        validation_loss.reset_states()
        validation_iou.reset_states()
    print("best validation iou:", best_iou)
    # 评价环节

    # 打印validation set里面的图片
    model_path = 'logs\\UnetPP_EM_SEG_20200515-213206\\best_model.h5'
    model.load_weights(model_path)
    valid_pred = model.predict(validation_data)
    for im_idx in range(valid_pred.shape[0]):
        valid_im = valid_pred[im_idx,:,:,:]
        valid_im[valid_im>=0.5] = 255
        valid_im[valid_im<0.5] = 0
        imsave(os.path.join(train_log_dir, 'data_%d.png'%im_idx), validation_data[im_idx])
        imsave(os.path.join(train_log_dir, 'validation_%d.png'%im_idx), valid_im)
        imsave(os.path.join(train_log_dir, 'validation_label_%d.png'%im_idx), validation_label[im_idx])
    # 生成测试集的输出
    test_pred = model.predict(test_data)
    test_tiff = []
    for im_idx in range(test_data.shape[0]):
        test_im = test_pred[im_idx,:,:,:]
        test_im[test_im>=0.5] = 255
        test_im[test_im<0.5] = 0
        test_tiff.append(np.squeeze(test_im))
    tiff = TIFFimage(test_tiff)
    tiff.write_file(os.path.join(train_log_dir, 'test_result.tif'))
train_EM_Seg()


