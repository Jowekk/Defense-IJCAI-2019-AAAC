import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize

from get_data import get_filename
from mynets import inception_resnet_v2
from mynets.mobilenet import mobilenet_v2
from mynets.resnet_v1 import resnet_v1
from tensorflow.contrib.slim.nets import inception, vgg
slim = tf.contrib.slim

import tensorflow as tf

FLAGS = tf.flags.FLAGS


CHECKPOINTS_DIR = './models/'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR,'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt'),
    'InceptionResnetV2': os.path.join(CHECKPOINTS_DIR, 'inception_resnet_v2', 'model.ckpt-231498'),
    'MobilenetV2': os.path.join(CHECKPOINTS_DIR, 'mobilenet_v2', 'model.ckpt-402416')}

max_epsilon = 32.0
num_iter = 40
batch_size = 11
momentum = 1.0


def preprocess_for_model(images, model_type):
    if 'inception' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224,224],align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224,224],align_corners=False)
        tmp_0 = images[:,:,:,0] - _R_MEAN
        tmp_1 = images[:,:,:,1] - _G_MEAN
        tmp_2 = images[:,:,:,2] - _B_MEAN
        images = tf.stack([tmp_0,tmp_1,tmp_2],3)
        return images

train_path, train_label = get_filename('./train_list.txt')
val_path, val_label = get_filename('./val_list.txt')

def load_input_images(batch_shape):
    filenames = []
    output_filenames = []
    input_images = np.zeros(batch_shape)
    idx = 0
    batch_size = batch_shape[0]
    for i in range(len(val_path)):
        path = val_path[i]
        output_path = path.replace("IJCAI_2019_AAAC_train", "IJCAI_fgsm_val")

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

def save_images(images, output_path):
    image = (((images + 1.0) * 0.5) * 255.0).astype(np.uint8)
        # resize back to [299, 299]
    image = imresize(image, [299, 299])
    Image.fromarray(image).save(output_path, format='JPEG')

def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_cam(grad_tensor, conv_tensor):
    weights = tf.reduce_mean(grad_tensor, axis=(1,2), keep_dims=True)
    cam_tensor = conv_tensor * weights
    cam_tensor = tf.image.resize_bilinear(cam_tensor, [224,224])
    cam_tensor = tf.reduce_sum(cam_tensor, axis=3)
    cam_tensor = tf.abs(cam_tensor)
    cam_max = tf.reduce_max(cam_tensor, axis=(1,2), keep_dims=True)
    cam_tensor = cam_tensor / cam_max
    return tf.expand_dims(cam_tensor, axis=3)

def non_target_graph(x, y, i, x_max, x_min, grad):

  eps = 2.0 * max_epsilon / 255.0
  alpha = eps / num_iter
  num_classes = 110

  with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
      x, num_classes=num_classes, is_training=False, scope='InceptionV1')

  y_inception = tf.reduce_sum(end_points_inc_v1["Logits"] * tf.one_hot(y, num_classes), axis=1)
  conv_grad_in = tf.gradients(y_inception, end_points_inc_v1["Mixed_5c"])[0]
  cam_inception = get_cam(conv_grad_in, end_points_inc_v1["Mixed_5c"])

  cam_mask = tf.to_float(tf.greater(cam_inception, (110.0/255.0)))
  cam_matrix = cam_mask * cam_inception
  cam_matrix = tf.stop_gradient(cam_matrix)

  inc_res_x = tf.image.resize_bilinear(x, [299, 299],align_corners=False)
  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_inc_res_v2, end_points_inc_res_v2 = inception_resnet_v2.inception_resnet_v2(
      inc_res_x, num_classes=num_classes, is_training=False, scope='InceptionResnetV2')

  with slim.arg_scope(mobilenet_v2.training_scope()):
    logits_mobi_v2, end_points_mobi_v2 = mobilenet_v2.mobilenet_v2_140(
      x, num_classes=num_classes, is_training=False, scope='MobilenetV2')
  end_points_mobi_v2['probs'] = tf.nn.softmax(logits_mobi_v2)

  # rescale pixle range from [-1, 1] to [0, 255] for resnet_v1 and vgg's input
  image = (((x + 1.0) * 0.5) * 255.0)
  processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
      processed_imgs_res_v1_50, num_classes=num_classes, is_training=False, scope='resnet_v1_50')

  end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
  end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

  # image = (((x + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
  processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
  with slim.arg_scope(vgg.vgg_arg_scope()):
    logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
      processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16')

  end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
  end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])


  # Using model predictions as ground truth to avoid label leaking
  pred = tf.argmax(end_points_inc_v1['Predictions'] 
                 + end_points_res_v1_50['probs'] 
                 + end_points_vgg_16['probs'] 
                 + end_points_inc_res_v2['Predictions']
                 + end_points_mobi_v2['probs'], 1)

  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)

  logits = (end_points_inc_v1['Logits']
          + end_points_res_v1_50['logits']
          + end_points_vgg_16['logits']
          + end_points_inc_res_v2['Logits']
          + logits_mobi_v2) / 5.0
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)

  Auxlogits = end_points_inc_res_v2['AuxLogits']
  Auxcross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  Auxlogits,
                                                  label_smoothing=0.0,
                                                  weights=0.4)

  noise = tf.gradients(cross_entropy + Auxlogits, x)[0]
  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = momentum * grad + noise
  x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise

def stop(x, y, i, x_max, x_min, grad):
  return tf.less(i, num_iter)

# Momentum Iterative FGSM
def non_target_mi_fgsm_attack():

  # some parameter
  eps = 2.0 * max_epsilon / 255.0
  batch_shape = [batch_size, 224, 224, 3]


  with tf.Graph().as_default():
    # Prepare graph
    raw_inputs = tf.placeholder(tf.uint8, shape=[None, 299, 299, 3])

    # preprocessing for model input,
    # note that images for all classifier will be normalized to be in [-1, 1]
    processed_imgs = preprocess_for_model(raw_inputs, 'inception_v1')

    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    y = tf.constant(np.zeros([batch_size]), tf.int64)
    # y = tf.placeholder(tf.int32, shape=[batch_size])
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape)
    x_adv, _, _, _, _, _ = tf.while_loop(stop, non_target_graph, [x_input, y, i, x_max, x_min, grad])

    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
    s4 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
    s5 = tf.train.Saver(slim.get_model_variables(scope='MobilenetV2'))

    with tf.Session() as sess:
      s1.restore(sess, model_checkpoint_map['inception_v1'])
      s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
      s3.restore(sess, model_checkpoint_map['vgg_16'])
      s4.restore(sess, model_checkpoint_map['InceptionResnetV2'])
      s5.restore(sess, model_checkpoint_map['MobilenetV2'])

      for filenames, input_images, output_filenames in load_input_images([batch_size, 299,299,3]):
        processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: input_images})
        adv_images = sess.run(x_adv, feed_dict={x_input: processed_imgs_})
        for k in range(len(filenames)):
            save_images(adv_images[k,:,:,:], output_filenames[k])
            print(output_filenames[k])
        
if __name__=='__main__':
     non_target_mi_fgsm_attack()
