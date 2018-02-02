from hparams import create_mixed_hparams
from network_utils import get_dataset
from model import model, model_arg_scope

import tensorflow as tf
import tensorflow.contrib.slim as slim

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='evaluate the accuracy')
parser.add_argument('--dataset_dir', default='', type=str, help='Path to the test data set.')
parser.add_argument('--checkpoint_dir', default='', type=str, help='Path where the checkpoint file locate.')
args = parser.parse_args()
num_classes = 18


def main():
    hparams = create_mixed_hparams()
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
                                scope=hparams.scope)
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
            num_corrects += int(sess.run(conf)[:])
            num_samples += 1
            print num_samples
    except tf.errors.OutOfRangeError:
        print 'epoch limit reached.'
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    print num_corrects * 1.0 / num_samples


if __name__ == '__main__':
    main()



