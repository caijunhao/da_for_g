from hparams import create_source_hparams
from network_utils import get_dataset
from model import model, model_arg_scope

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

images_t = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])

with slim.arg_scope(model_arg_scope()):
    net, end_points = model(inputs=images_t,
                            num_classes=num_classes,
                            is_training=False,
                            dropout_keep_prob=1.0,
                            scope='source_only')
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


