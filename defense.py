#########################################################################################################################################
#   name                    model              dataset    filter   Teacher     score      test accuracy     NOTICE
# fgsm_Defense        inception resnet v2       fgsm        N          Y       +0.07                          TODO
#  ATN_Defense        inception resnet v2       ATN         N          Y       -0.01      
#  adv_Defense        inception resnet v2    fgsm + ATN     Y          Y
#CAM_Defense_in_re_v2 inception resnet v2        CAM        N          Y       -0.34/2
#  Defense_vgg              vgg 16               fgsm       Y          Y       -0.13/2
#Defense_Inception_v4_fgsm   inception v4        fgsm       Y          Y       -0.34/2
#
#InceptionResnetV2    inception resnet v2        fgsm       Y          N 
#  resnet_v2_152           resnet v2             fgsm       Y          N       0.6634
#   inception             inception V1           fgsm       Y          N       +0.03        0.57845
#  MobilenetV2            MobilenetV2            fgsm       Y          N                    0.60855
#  InceptionV4            inception v4           fgsm       Y          N       +0.29 ??     0.65525
#     VGG_19                 vgg 19              fgsm       Y          N       -0.02        0.55355
# fgsm_inception_v3       inception v3           fgsm       Y          N       -0.09        0.61545          TODO submit

#Ten_Filter_Defense   inception resnet v2        fgsm      Ten         N       +0.41        0.6136
#  resnet_v1_101         resnet v1 101           fgsm      Ten         N       +0.18        0.57005
#     VGG_16                 vgg 16              fgsm      Ten         N       +0.13        0.5029           TODO next training
#ten_filter_inception_v4  inception v4           fgsm      Ten         N       +0.14        0.53835          TODO next training, already
#  resnet_v1_50           resnet v1 50           fgsm      Ten         N       +0.03???     0.55865
#Ten_filter_MobilenetV2   mobilenet v2 140       fgsm      Ten         N       -0.03??      0.5874           

#  InceptionV3            InceptionV3          rand cam     Y          N       +0.248       0.6405           15.6540
#  resnet_v1_152          resnet v1 152        rand cam     Y          N       +0.13        0.6509
#filter_rand_resnet_50      resnet 50          rand cam     Y          N       -0.01        0.6506           TODO submit
#filter_rand_in_re_v2  inception resnet v2     rand cam     Y          N       -0.05  XX, don't use this

#  Mask_attack        inception resnet v2        mask       Y          N       +0.19        0.60975
#  Inception_v2           Inception v2           mask       Y          N       +0.04        0.5759

#edge_in_re_v2   inception resnet v2        Edge preserve   N          N       +0.19        0.6994
#edge_resnet_v1_152    resnet v1 152        Edge preserve   N          N       +0.05        0.64305
#edge_inception_v4    inception v4          Edge preserve   N          N       -0.03        0.6118 



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random
import numpy as np
import tensorflow as tf

from flags import *
from preprocessing import tf_preprocessing
from mynets import inception_resnet_v2, inception_v4, inception_v3, inception_v2, vgg
from mynets.resnet import resnet_v2, resnet_v1
from mynets.mobilenet import mobilenet_v2
from tensorflow.contrib.slim import nets
from load_and_save_imgs import load_images
#from filters import gauss_filter
from scipy.ndimage import filters
slim = tf.contrib.slim

def edge_preserving(A):
    B = np.zeros((A.shape), dtype=np.float32)
    for i in range(A.shape[0]):
        B[i,:,:,:] = cv2.edgePreservingFilter(A[i,:,:,:], flags=1, sigma_s=30, sigma_r=0.6)
    return B

def gauss_filter(A, r):
    output_imgs = np.zeros((A.shape), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(3):
            output_imgs[i, :, :, j] = filters.gaussian_filter(A[i, :, :, j], r)
    return output_imgs

def compute_pred_val(logits):
    labels_list = list()
    preds = np.argmax(logits, 1)
    for i in range(preds.shape[0]):
        labels_list.append(preds[i])
    return labels_list

def main(_):
    batch_shape = [8, 299, 299, 3]
    nb_classes = FLAGS.num_classes

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape) 
        five_input = tf.placeholder(tf.float32, shape=batch_shape) 
        ten_input = tf.placeholder(tf.float32, shape=batch_shape) 

        is_training = tf.placeholder(tf.bool, [])

        edge_tensor = tf_preprocessing(x_input, 'inception', 299,299) 
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            edge_logits, edge_end_points = inception_resnet_v2.inception_resnet_v2(
                      edge_tensor, num_classes=nb_classes, is_training=False, scope='edge_in_re_v2') 

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_edge_ince_v4, _ = inception_v4.inception_v4(
                  edge_tensor, num_classes=nb_classes, is_training=False, scope='edge_inception_v4')

        edge_res_tensor = tf_preprocessing(x_input, 'vgg', 224,224) 
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            edge_logits_res_v1, _ = resnet_v1.resnet_v1_152(
                      edge_res_tensor, num_classes=110, is_training=True, scope='edge_resnet_v1_152') 


        defense_tensor = tf_preprocessing(ten_input, 'inception', 299,299) 
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            defense_Ten_logits, defense_Ten_end_points = inception_resnet_v2.inception_resnet_v2(
                      defense_tensor, num_classes=nb_classes, is_training=False, scope='Ten_Filter_Defense')

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_Ten_ince_v4, Ten_in_v4_end_points = inception_v4.inception_v4(
                  defense_tensor, num_classes=nb_classes, is_training=False, scope='ten_filter_inception_v4')

        mobi_Ten_tensor = tf_preprocessing(ten_input, 'inception', 224,224) 
        with slim.arg_scope(mobilenet_v2.training_scope()):
            logits_Ten_mobi, _ = mobilenet_v2.mobilenet_v2_140(
                  mobi_Ten_tensor, num_classes=nb_classes, is_training=True, scope='Ten_filter_MobilenetV2')

        res_v1_101_tensor = tf_preprocessing(ten_input, 'vgg', 224,224) 
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits_res_v1_101, _ = resnet_v1.resnet_v1_101(
              res_v1_101_tensor, num_classes=110, is_training=False, scope='resnet_v1_101')

        with slim.arg_scope(vgg.vgg_arg_scope()):
            vgg_16_logits, _ = vgg.vgg_16(
                  res_v1_101_tensor, num_classes=110, is_training=False, scope='vgg_16')

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits_Ten_res_v1_50, _ = resnet_v1.resnet_v1_50(
              res_v1_101_tensor, num_classes=110, is_training=True, scope='resnet_v1_50')


        in_tensor = tf_preprocessing(five_input, 'inception', 299,299) 
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            defense_fgsm_logits, defense_fgsm_end_points = inception_resnet_v2.inception_resnet_v2(
                      in_tensor, num_classes=nb_classes, is_training=True, scope='fgsm_Defense')

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            mask_logits, mask_end_points = inception_resnet_v2.inception_resnet_v2(
                      in_tensor, num_classes=nb_classes, is_training=True, scope='Mask_attack')

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_inc_res_v2, _ = inception_resnet_v2.inception_resnet_v2(
              in_tensor, num_classes=110, is_training=is_training, scope='InceptionResnetV2')

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_ince_v4, in_v4_end_points = inception_v4.inception_v4(
                  in_tensor, num_classes=nb_classes, is_training=True, scope='InceptionV4')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ince_v3, in_v3_end_points = inception_v3.inception_v3(
                  in_tensor, num_classes=nb_classes, is_training=True, scope='InceptionV3')

        res_tensor = tf_preprocessing(five_input, 'vgg', 224,224) 
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_res_v2, _ = resnet_v2.resnet_v2_152(
              res_tensor, num_classes=110, is_training=False, scope='resnet_v2_152')

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits_res_v1, _ = resnet_v1.resnet_v1_152(
              res_tensor, num_classes=110, is_training=False, scope='resnet_v1_152')


        mobi_tensor = tf_preprocessing(five_input, 'inception', 224,224) 
        with slim.arg_scope(mobilenet_v2.training_scope()):
            logits_mobi, _ = mobilenet_v2.mobilenet_v2_140(
                  mobi_tensor, num_classes=nb_classes, is_training=False)

        with slim.arg_scope(nets.inception.inception_v1_arg_scope()):
            logits_ince, endp = nets.inception.inception_v1( 
                    mobi_tensor, num_classes=nb_classes, is_training=False)

        with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
            logits_in_v2, _ = inception_v2.inception_v2( 
                    mobi_tensor, num_classes=nb_classes, is_training=False)

        logits_tensor = (logits_inc_res_v2 * 1.5
                       + logits_res_v2
                       + logits_mobi
                       + defense_fgsm_logits
                       + logits_ince_v4
                       + defense_Ten_logits * 2.0
                       + mask_logits
                       + logits_ince_v3 * 1.5
                       + logits_res_v1
                       + logits_in_v2
                       + logits_res_v1_101
                       + edge_logits * 1.8
                       + edge_logits_res_v1
                       + vgg_16_logits
                       + logits_Ten_ince_v4
                       + logits_Ten_res_v1_50
                       + logits_edge_ince_v4) / 17.0 

        s = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_101'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='MobilenetV2'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_152'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='InceptionV2'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='edge_resnet_v1_152'))
        s10 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
        s12 = tf.train.Saver(slim.get_model_variables(scope='ten_filter_inception_v4'))
        s13 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s14 = tf.train.Saver(slim.get_model_variables(scope='Ten_filter_MobilenetV2'))
        s15 = tf.train.Saver(slim.get_model_variables(scope='edge_inception_v4'))

        defense_fgsm_saver = tf.train.Saver(slim.get_model_variables(scope='fgsm_Defense'))
        defense_mask_saver = tf.train.Saver(slim.get_model_variables(scope='Mask_attack'))
        defense_Ten_saver = tf.train.Saver(slim.get_model_variables(scope='Ten_Filter_Defense'))
        defense_edge_saver = tf.train.Saver(slim.get_model_variables(scope='edge_in_re_v2'))


        # Run computation
        with tf.Session() as sess:

            s.restore(sess, "./mymodels/resnet_v2/resnet_v2.ckpt")
            s1.restore(sess, "./mymodels/inception_v1/inception_v1.ckpt")
            s2.restore(sess, "./mymodels/inception_v4/inception_v4.ckpt")
            s3.restore(sess, "./mymodels/resnet_v1_101/resnet_v1_101.ckpt")
            s4.restore(sess, "./mymodels/InceptionResnetV2/InceptionResnetV2.ckpt")
            s5.restore(sess, "./mymodels/mobilenet_v2_140/mobilenet_v2_140.ckpt")
            s6.restore(sess, "./mymodels/inception_v3/inception_v3.ckpt")
            s7.restore(sess, "./mymodels/resnet_v1_152/resnet_v1_152.ckpt")
            s8.restore(sess, "./mymodels/inception_v2/inception_v2.ckpt") 
            s9.restore(sess, "./mymodels/edge_resnet_v1_152/edge_resnet_v1_152.ckpt") 
            s10.restore(sess, "./mymodels/ten_filter_vgg16/vgg_16.ckpt") 
            s12.restore(sess, "./mymodels/ten_filter_inception_v4/ten_filter_inception_v4.ckpt") 
            s13.restore(sess, "./mymodels/ten_filter_resnet_v1_50/ten_filter_resnet_v1_50.ckpt") 
            s14.restore(sess, "./mymodels/ten_filter_mobilenet/ten_filter_mobilenet.ckpt") 
            s15.restore(sess, "./mymodels/edge_inception_v4/edge_inception_v4.ckpt") 

            defense_fgsm_saver.restore(sess, "./mymodels/fgsm_defense/defense.ckpt-82000")
            defense_Ten_saver.restore(sess, "./mymodels/ten_filter_defense/ten_filter.ckpt") 
            defense_mask_saver.restore(sess, "./mymodels/mask_attack/mask.ckpt") 
            defense_edge_saver.restore(sess, "./mymodels/edge_in_re_v2/edge.ckpt") 

            #s15.save(sess, "./mymodels/edge_inception_v4.ckpt")

            with open(FLAGS.output_file, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    edge_imgs = edge_preserving(images)
                    five_filter = gauss_filter(images, 5)
                    ten_filter = gauss_filter(images, 10)
                    logits_val = sess.run(logits_tensor, feed_dict={five_input: five_filter, 
                                                                    ten_input: ten_filter, 
                                                                    x_input: edge_imgs,
                                                                    is_training:True}) 
                    labels = compute_pred_val(logits_val)
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))



if __name__ == '__main__':
    tf.app.run()
