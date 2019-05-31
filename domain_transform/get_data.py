import os
import numpy as np
import tensorflow as tf

def get_filename(filename):
    y = list()
    path_list = list()
    filename_list = open(filename)
    for line in filename_list:
        line=line.strip('\n')
        tmp_name = line.split('/')
        y.append(int(tmp_name[6])) # 6
        path_list.append(line)
    return path_list, y


def get_batch(image, label, image_W, image_H, batch_size, capacity=2000):
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    

    image = tf.image.resize_images(image, [299, 299])
    image = tf.random_crop(image, [image_W, image_H, 3])
    #image = tf.image.random_flip_left_right(image)
    
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity,
                                                min_after_dequeue=500)

    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    #image_batch = tf.image.resize_bilinear(image_batch, [image_W, image_H],align_corners=False)
    
    return image_batch, label_batch

