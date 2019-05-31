from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy.ndimage import filters, median_filter

means = [123.68, 116.779, 103.939]

def vgg_preprocessing(image):
    num_channels = image.shape[-1]
    if num_channels != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    return (image - means)

def inception_preprocessing(image):
    image = image / 255.0
    image = image - 0.5
    image = image * 2
    return image

def preprocessing_for_eval(image, model_name):
    if (model_name == "inception" or model_name == "inception_resnet_v2"):
        return inception_preprocessing(image)
    elif(model_name == "resnet" or model_name == "vgg"):
        return vgg_preprocessing(image)
    else:
        raise RuntimeError('Undefined model name')

def restore_for_show(image, model_name):
    image = tf.image.resize_bilinear(image, [299, 299],align_corners=False)
    if (model_name == "inception" or model_name == "inception_resnet_v2"):
        return (image / 2.0 + 0.5) * 255.0
    elif(model_name == "resnet" or model_name == "vgg"):
        return image + means
    else:
        raise RuntimeError('Undefined model name')


def tf_preprocessing(image, model_name, height, width):
    image = tf.image.resize_bilinear(image, [height, width],align_corners=False)
    if (model_name == "inception" or model_name == "inception_resnet_v2"):
        image = tf.divide(image, 255.0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2)
        return image
    elif(model_name == "resnet" or model_name == "vgg"):
        num_channels = image.get_shape().as_list()[-1]
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=3, values=channels)
    else:
        raise RuntimeError('Undefined model name')

def gauss_filter(A):
    output_imgs = np.zeros((A.shape), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(3):
            A[i, :, :, j] = median_filter(A[i, :, :, j], (10,10))
            output_imgs[i, :, :, j] = filters.gaussian_filter(A[i, :, :, j], 8)
    return output_imgs


