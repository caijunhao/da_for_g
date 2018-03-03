from hparams import *
from network_utils import get_dataset
from model import *

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='evaluate the accuracy')
parser.add_argument('--checkpoint_dir', default='', type=str, help='Path where the checkpoint file locate.')
args = parser.parse_args()
num_classes = 18

hparams = create_semi_supervised_domain_adapt_hparams()

images_t = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
images_t = images_t - [123.68, 116.779, 103.939]

with slim.arg_scope(model_arg_scope()):
    net, end_points = model(inputs=images_t,
                            num_classes=num_classes,
                            is_training=False,
                            dropout_keep_prob=1.0,
                            reuse=tf.AUTO_REUSE,
                            scope=hparams.scope,
                            adapt_scope='target_adapt_layer',
                            adapt_dims=128)
    angle_index = tf.argmin(net, axis=1)

    saver = tf.train.Saver()
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
    print 'Successfully loading model.'

    images = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    print 'angle: %d' % sess.run(angle_index, feed_dict={images_t: images})


