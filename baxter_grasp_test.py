from hparams import *
from network_utils import get_dataset
from model import *

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from math import pi

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


parser = argparse.ArgumentParser(description='grasping')
parser.add_argument('--topic_name', default='/cameras/right_hand_camera/image', type=str)
parser.add_argument('--patch_size', default=224, type=int, help='image patch size to crop')
parser.add_argument('--num_patches', default=400, type=int)
parser.add_argument('--dataset_dir', default='', type=str, help='Path to the test data set.')
parser.add_argument('--checkpoint_dir', default='', type=str, help='Path where the checkpoint file locate.')
args = parser.parse_args()
num_classes = 18


def process_and_draw_rect(img, grasp_angle, sx, sy, flag=1):

    grasp_l = 100 / 3.0
    grasp_w = 100 / 6.0
    points = np.array([[-grasp_l, -grasp_w],
                       [grasp_l, -grasp_w],
                       [grasp_l, grasp_w],
                       [-grasp_l, grasp_w]])
    rotate_matrix = np.array([[np.cos(grasp_angle), -np.sin(grasp_angle)],
                              [np.sin(grasp_angle), np.cos(grasp_angle)]])
    rot_points = np.dot(rotate_matrix, points.transpose()).transpose()
    temp = np.array([[sx, sy],
                     [sx, sy],
                     [sx, sy],
                     [sx, sy]])
    im_points = (rot_points + temp).astype(np.int)
    if flag:
        cv2.line(img, tuple(im_points[0]), tuple(im_points[1]), color=(0, 255, 0), thickness=5)
        cv2.line(img, tuple(im_points[1]), tuple(im_points[2]), color=(0, 0, 255), thickness=5)
        cv2.line(img, tuple(im_points[2]), tuple(im_points[3]), color=(0, 255, 0), thickness=5)
        cv2.line(img, tuple(im_points[3]), tuple(im_points[0]), color=(0, 0, 255), thickness=5)
    else:
        cv2.line(img, tuple(im_points[0]), tuple(im_points[1]), color=(0, 255, 0), thickness=5)
        cv2.line(img, tuple(im_points[1]), tuple(im_points[2]), color=(255, 0, 0), thickness=5)
        cv2.line(img, tuple(im_points[2]), tuple(im_points[3]), color=(0, 255, 0), thickness=5)
        cv2.line(img, tuple(im_points[3]), tuple(im_points[0]), color=(255, 0, 0), thickness=5)
    return img


def main():
    hparams = create_semi_supervised_domain_adapt_hparams()
    images, class_labels, theta_labels = get_dataset(args.dataset_dir,
                                                     num_readers=1,
                                                     num_preprocessing_threads=1,
                                                     hparams=hparams,
                                                     shuffle=False,
                                                     num_epochs=1,
                                                     is_training=False)

    with slim.arg_scope(model_arg_scope()):
        net, end_points = model(inputs=images,
                                num_classes=num_classes,
                                is_training=False,
                                dropout_keep_prob=1.0,
                                reuse=tf.AUTO_REUSE,
                                scope=hparams.scope,
                                adapt_scope='target_adapt_layer',
                                adapt_dims=128)
        angle_index = tf.argmin(net, axis=1)
        theta_lebels_one_hot = tf.one_hot(theta_labels, depth=18, on_value=1.0, off_value=0.0)
        theta_acted = tf.reduce_sum(tf.multiply(net, theta_lebels_one_hot), axis=1, name='theta_acted')
        sig_op = slim.nn.sigmoid(theta_acted)
        conf = tf.equal(tf.to_int32(tf.greater_equal(sig_op, 0.5)),
                        tf.to_int32(tf.greater_equal(tf.to_float(class_labels), 0.1)))

    saver = tf.train.Saver()
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
    print 'Successfully loading model: {}.'.format(tf.train.latest_checkpoint(args.checkpoint_dir))
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_corrects = 0
    num_samples = 0
    try:
        while not coord.should_stop():
            img, cls, theta, angle, con, score= sess.run([images, class_labels, theta_labels, angle_index, conf, net])
            img = np.squeeze(img+np.array([123.68, 116.779, 103.939])).astype(np.uint8)
            cls = cls[0]
            theta = theta[0]
            angle = angle[0]
            grasp_angle = (angle * 10 - 90) * 1.0 / 180 * pi
            theta_angle = (theta * 10 - 90) * 1.0 / 180 * pi
            con = con[0]
            img = process_and_draw_rect(img, grasp_angle, 112, 112)
            img = process_and_draw_rect(img, theta_angle, 112, 112, 0)
            num_corrects += int(con)
            num_samples += 1
            print img.shape, cls, theta, angle, con
            print score
            cv2.imshow('img', img[..., ::-1])
            cv2.waitKey(0)
    except tf.errors.OutOfRangeError:
        print 'epoch limit reached.'
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()

