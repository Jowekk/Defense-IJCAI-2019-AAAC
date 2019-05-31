from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import csv
import shutil
import numpy as np
import tensorflow as tf

from flags import *
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with open(filepath) as f:
            raw_image = imread(f, mode='RGB')
            image = imresize(raw_image, [299, 299]).astype(np.float)
            #image = (image / 255.0) * 2.0 - 1.0
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

'''
# load input images to generate attack images
def load_input_images(input_dir, batch_shape):
    filenames = []
    input_images = np.zeros(batch_shape)
    idx = 0
    batch_size = batch_shape[0]
    with open(os.path.join(input_dir, 'dev.csv'), 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:

            filepath = os.path.join(input_dir, row['filename'])
            with open(filepath) as f:
                raw_image = imread(f, mode='RGB').astype(np.float)

            input_images[idx, :, :, :] = raw_image
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, input_images
                filenames = []
                input_images = np.zeros(batch_shape)
                idx = 0
        if idx > 0:
            yield filenames, input_images
'''
