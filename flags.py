from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.flags.DEFINE_string(
    'model_name', '', 'model name.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_file', '', 'Output result.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')
FLAGS = tf.flags.FLAGS

if FLAGS.model_name == "inception_resnet_v2":
    FLAGS.image_height = 299
    FLAGS.image_width = 299
else:
    FLAGS.image_height = 224
    FLAGS.image_width = 224

#FLAGS.model_name = "inception"
