from hparams import create_target_hparams
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
    hparams = create_target_hparams()
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
    num_pos, num_neg = 0.0, 0.0
    correct_pos, correct_neg = 0.0, 0.0
    try:
        while not coord.should_stop():
            con, label = sess.run([conf, class_labels])
            con = int(con[:])
            label = label[:]
            if label:
                num_neg += 1
                correct_neg += con
            else:
                num_pos += 1
                correct_pos += con
    except tf.errors.OutOfRangeError:
        print 'epoch limit reached.'
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    tp = correct_pos / num_pos
    tn = correct_neg / num_neg
    fp = (num_pos - correct_pos) / num_pos
    fn = (num_neg - correct_neg) / num_neg
    print 'num_pos: {}, num_neg: {}, correct_pos: {}, correct_neg: {}'.format(num_pos, num_neg, correct_pos,
                                                                              correct_neg)
    print 'tp, tn, fp, fn: {}, {}, {}, {}'.format(tp, tn, fp, fn)
    print 'precision: {}'.format(tp * 1.0 / (tp + fp))
    print 'recall: {}'.format(tp * 1.0 / (tp + fn))
    print 'F1 score: {}'.format(2.0 * tp / (2.0 * tp + fp + tn))
    print 'accuracy: {}'.format((correct_neg + correct_pos) / (num_neg + num_pos))


if __name__ == '__main__':
    main()



