from model import *
from hparams import *

import dataset_utils

import tensorflow as tf
import numpy as np
import PIL.Image
import random
from math import pi

import argparse
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


parser = argparse.ArgumentParser(description='create tf record')
parser.add_argument('--set', default='Target4', type=str, help='Convert training set, validation set or test set.')
parser.add_argument('--data_dir', default='/data/cmu_patch_datasets', type=str, help='Path of dataset.')
parser.add_argument('--checkpoint_dir1', default='', type=str, help='Path where the checkpoint file locate.')
parser.add_argument('--checkpoint_dir2', default='', type=str, help='Path where the checkpoint file locate.')
parser.add_argument('--output_path', default='', type=str, required=True, help='Path of record.')
parser.add_argument('--image_size', default=224, type=int, help='Image size.')
args = parser.parse_args()

sets = ['Train', 'Test', 'Validation']  # source domain data set with argumentation.
sets.extend(['Target0', 'Target1', 'Target2', 'Target3', 'Target4', 'Target5', 'Target6'])
sets.extend(['t1', 't2', 't3', 't4', 't5', 't6'])  # modified target domain data set with argumentation.
sets.extend(['t0'])  # target domain with argumentation.
labels = ['positive', 'negative']
label_file = 'dataInfo.txt'
folder = 'Images'
num_classes = 18


def convert_theta(theta):
    # theta : a num from -pi/2 to pi/2
    # return : a num from 0 to 17
    if theta > pi / 2:
        theta = theta - pi
    if theta < -pi / 2:
        theta = theta + pi
    theta = (theta + pi / 2) * 180 / pi  # [0,pi]
    diff = [abs(theta - i * 10) for i in xrange(18)]
    return diff.index(min(diff))


def dict_to_tf_example(encoded_jpg, label, theta):
    class_label = 0 if label == 'positive' else 1
    theta_label = theta
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_utils.bytes_feature(encoded_jpg),
        'image/format': dataset_utils.bytes_feature('jpeg'),
        'image/class/label': dataset_utils.int64_feature(class_label),
        'image/theta/label': dataset_utils.int64_feature(theta_label),
    }))
    return example


def main():
    if args.set not in sets:
        raise ValueError('set must be in : {}'.format(sets))

    data_dir = os.path.join(args.data_dir, args.set)

    domain_adapt_hparams = create_domain_adapt_hparams()
    images_t_1 = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    images_t_1 = images_t_1 - [123.68, 116.779, 103.939]
    with slim.arg_scope(model_arg_scope()):
        net_1, _ = model(inputs=images_t_1,
                         num_classes=num_classes,
                         is_training=False,
                         dropout_keep_prob=1.0,
                         reuse=tf.AUTO_REUSE,
                         scope=domain_adapt_hparams.scope,
                         adapt_scope='target_adapt_layer',
                         adapt_dims=128)
    min_index_1 = tf.argmin(net_1, axis=1)
    max_index_1 = tf.argmax(net_1, axis=1)
    variables_to_resotre_1 = slim.get_model_variables(domain_adapt_hparams.scope)
    saver1 = tf.train.Saver(variables_to_resotre_1)

    semi_supervised_hparams = create_semi_supervised_domain_adapt_hparams()
    images_t_2 = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    images_t_2 = images_t_2 - [123.68, 116.779, 103.939]
    with slim.arg_scope(model_arg_scope()):
        net_2, _ = model(inputs=images_t_2,
                         num_classes=num_classes,
                         is_training=False,
                         dropout_keep_prob=1.0,
                         reuse=tf.AUTO_REUSE,
                         scope=semi_supervised_hparams.scope,
                         adapt_scope='target_adapt_layer',
                         adapt_dims=128)
    min_index_2 = tf.argmin(net_2, axis=1)
    max_index_2 = tf.argmax(net_2, axis=1)
    variables_to_resotre_2 = slim.get_model_variables(semi_supervised_hparams.scope)
    saver2 = tf.train.Saver(variables_to_resotre_2)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    saver1.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir1))
    saver2.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir2))
    print 'Successfully loading model: {}.'.format(tf.train.latest_checkpoint(args.checkpoint_dir1))
    print 'Successfully loading model: {}.'.format(tf.train.latest_checkpoint(args.checkpoint_dir2))

    pos_writer = tf.python_io.TFRecordWriter(os.path.join(args.output_path,
                                                          'multi_model_pseudo_positive'+'_'+args.set+'.tfrecord'))
    neg_writer = tf.python_io.TFRecordWriter(os.path.join(args.output_path,
                                                          'multi_model_pseudo_negative'+'_'+args.set+'.tfrecord'))
    for label in labels:
        data_info = dataset_utils.read_examples_list(os.path.join(data_dir, label, label_file))
        random.shuffle(data_info)
        for image_name, theta in data_info:
            image_path = os.path.join(data_dir, label, folder, image_name)
            with tf.gfile.GFile(image_path, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            image = image.resize((224, 224))
            image_n = np.array(image)
            image_n = np.expand_dims(image_n, axis=0).astype(np.float32)
            scores_1, min_idx_1, max_idx_1 = sess.run([net_1, min_index_1, max_index_1],
                                                      feed_dict={images_t_1: image_n})
            scores_2, min_idx_2, max_idx_2 = sess.run([net_2, min_index_2, max_index_2],
                                                      feed_dict={images_t_2: image_n})
            if np.min(scores_1) < 0 and np.max(scores_1) > 0 and np.min(scores_2) < 0 and np.max(scores_2) > 0:
                if np.min(scores_1) < -5.0 and np.min(scores_1) < np.min(scores_2):
                    tf_example = dict_to_tf_example(encoded_jpg, 'positive', min_idx_1[0])
                    pos_writer.write(tf_example.SerializeToString())
                elif np.min(scores_2) < -5.0 and np.min(scores_2) < np.min(scores_1):
                    tf_example = dict_to_tf_example(encoded_jpg, 'positive', min_idx_2[0])
                    pos_writer.write(tf_example.SerializeToString())
                if np.max(scores_1) > 5.0 and np.max(scores_1) > np.max(scores_2):
                    tf_example = dict_to_tf_example(encoded_jpg, 'negative', max_idx_1[0])
                    neg_writer.write(tf_example.SerializeToString())
                elif np.max(scores_2) > 5.0 and np.max(scores_2) > np.max(scores_1):
                    tf_example = dict_to_tf_example(encoded_jpg, 'negative', max_idx_2[0])
                    neg_writer.write(tf_example.SerializeToString())
    pos_writer.close()
    neg_writer.close()


if __name__ == '__main__':
    main()
