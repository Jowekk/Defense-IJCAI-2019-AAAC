from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imresize
from get_data import get_filename
from PIL import Image

from models import get_model
#from load_and_save_imgs import save_images
from preprocessing import tf_preprocessing, tf_restore 
slim = tf.contrib.slim

CHECKPOINTS_DIR = './models/'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR,'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt'),
    'InceptionResnetV2': os.path.join(CHECKPOINTS_DIR, 'inception_resnet_v2', 'model.ckpt-231498'),
    'alexnet_v2': os.path.join(CHECKPOINTS_DIR, 'alexnet_v2', 'model.ckpt-463663'),
    'MobilenetV2': os.path.join(CHECKPOINTS_DIR, 'mobilenet_v2', 'model.ckpt-402416')}

train_path, train_label = get_filename('./train_list.txt')
val_path, val_label = get_filename('./val_list.txt')

def load_input_images(batch_shape):
    filenames = []
    output_filenames = []
    input_images = np.zeros(batch_shape)
    idx = 0
    batch_size = batch_shape[0]
    for i in range(34626, len(train_path)):
        path = train_path[i]
        output_path = path.replace("IJCAI_2019_AAAC_train", "IJCAI_cam_train")

        out_path_dir = os.path.split(output_path)[0] + "/"
        if not os.path.exists(out_path_dir):
            os.makedirs(out_path_dir)

        img = Image.open(path)
        try:
            img = img.resize((299, 299),Image.ANTIALIAS)
        except:
            continue

        input_images[idx, :, :, :] = img
        filenames.append(path)
        output_filenames.append(output_path)
        idx += 1
        if idx == batch_size:
            yield filenames, input_images, output_filenames
            filenames = []
            output_filenames = []
            input_images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, input_images, output_filenames


def save_images(image, output_path):
    #image = (((images + 1.0) * 0.5) * 255.0).astype(np.uint8)
        # resize back to [299, 299]
    #image = imresize(image, [299, 299])
    Image.fromarray(image).save(output_path, format='JPEG')


def get_cam(conv_data, grad_data):
    weights = np.mean(grad_data, axis = (0,1))
    cam_data = np.zeros(conv_data.shape[0:2], np.float32)
    for i, w in enumerate(weights):
        cam_data = cam_data + w * conv_data[:,:,i]
    cam_data = abs(cam_data)
    cam_data = imresize(cam_data,[224,224])
    cam_data = cam_data / np.max(cam_data)
    #cam_data[cam_data < 0.3] = 0.0
    #cam_data[cam_data > 0.8] = 0.8
    cam_data = cam_data * np.random.rand()
    return cam_data

def get_cam_matrix(conv_matrix, grad_matrix):
    cam_matrix = np.zeros((conv_matrix.shape[0], 224,224,3))
    for i in range(conv_matrix.shape[0]):
        conv_val = conv_matrix[i,:,:,:]
        grad_val = grad_matrix[i,:,:,:]
        for j in range(3):
            cam_matrix[i,:,:,j] = get_cam(conv_val, grad_val)
    return cam_matrix


def cam_attack(mylist, sess, conv_tensor, conv_grad, x_adv):
    filenames = list()
    imglist = list()
    labelist = list()
    for i in range(len(mylist)):
        filenames.append(mylist[i][0])
        imglist.append(mylist[i][1][np.newaxis,:,:,:])
        labelist.append(mylist[i][2])
    input_images = np.concatenate(imglist, axis=0)
    
    conv_val, conv_grad_val = sess.run([conv_tensor, conv_grad], 
                      feed_dict={x_input: input_images, labels: labelist})
    conv_matrix = conv_val
    cam_matrix = get_cam_matrix(conv_matrix, conv_grad_val)
    x_adv_val = sess.run(x_adv, 
        feed_dict={x_input: input_images, labels: labelist, eps_matrix:cam_matrix})
    new_list = list()
    for i in range(len(filenames)):
        new_list.append([filenames[i], x_adv_val[i,:,:,:], labelist[i]])

    return new_list

def get_adv(model_name, x_input, labels, eps_matrix):
    if model_name == 'inception':
        size = 224
        logit_name = "Logits"
        conv_name = "Mixed_5c"
    if model_name == 'resnet':
        size = 224
        logit_name = "resnet_v1_50/logits"
        conv_name = "resnet_v1_50/block4/unit_3/bottleneck_v1"
    if model_name == 'vgg':
        size = 224
        logit_name = "vgg_16/fc8"
        conv_name = "vgg_16/conv5/conv5_3"

    x_input = tf.image.resize_bilinear(x_input, [size, size],align_corners=False)
    net_input = tf_preprocessing(x_input, model_name)
    logits, end_points = get_model(net_input, 110, model_name)
    conv_tensor = end_points[conv_name]
    
    y_c = tf.reduce_sum(tf.multiply(end_points[logit_name], tf.one_hot(labels, nb_classes)), axis=1)
    conv_grad = tf.gradients(y_c, conv_tensor)[0]
    cross_entropy = tf.losses.softmax_cross_entropy(tf.one_hot(labels, nb_classes),
                                                    logits,
                                                    label_smoothing=0.1)

    delta_matrix = eps_matrix * tf.sign(tf.gradients(cross_entropy, net_input)[0])
    x_adv = net_input + delta_matrix
    x_adv = tf_restore(x_adv, model_name)
    x_adv = tf.image.resize_bilinear(x_adv, [299, 299],align_corners=False)
    x_adv = tf.clip_by_value(x_adv, 0, 255)
    return conv_tensor, conv_grad, x_adv

def model_preds(x, name, size):
    x_tensor = tf_preprocessing(x_input, name, size, size)
    logits, _ = get_model(x_tensor, 110, name)
    preds = tf.nn.softmax(logits)
    return preds

nb_classes = 110
batch_shape = [32, 299, 299, 3]

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
    eps_matrix = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    labels = tf.placeholder(tf.int32, shape=[None])

    # cam attack
    inception_conv_tensor, inception_conv_grad, inception_x_adv = get_adv('inception', x_input, labels, eps_matrix)
    resnet_conv_tensor, resnet_conv_grad, resnet_x_adv = get_adv('vgg', x_input, labels, eps_matrix)
    vgg_conv_tensor, vgg_conv_grad, vgg_x_adv = get_adv('resnet', x_input, labels, eps_matrix)

    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))


    with tf.Session() as sess:
        s1.restore(sess, model_checkpoint_map['inception_v1'])
        s3.restore(sess, model_checkpoint_map['vgg_16'])
        s2.restore(sess, model_checkpoint_map['resnet_v1_50'])

        for filenames, input_images, output_filenames in load_input_images([32, 299,299,3]):
            new_list = list()
            for i in range(len(filenames)):
                trlabels = filenames[i].split('/')[6]
                new_list.append([filenames[i], input_images[i,:,:,:], trlabels])

            new_list = cam_attack(new_list, sess, inception_conv_tensor, inception_conv_grad, inception_x_adv)
            new_list = cam_attack(new_list, sess, resnet_conv_tensor, resnet_conv_grad, resnet_x_adv)
            mylist = cam_attack(new_list, sess, vgg_conv_tensor, vgg_conv_grad, vgg_x_adv)

            imglist = list()
            for i in range(len(mylist)):
                #savenames.append(mylist[i][0])
                imglist.append(mylist[i][1][np.newaxis,:,:,:])
            saveimgs = np.concatenate(imglist, axis=0)

            for k in range(len(filenames)):
                save_images(saveimgs[k,:,:,:].astype(np.uint8), output_filenames[k])
                print(output_filenames[k])

            
