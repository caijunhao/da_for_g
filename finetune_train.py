from hparams import create_finetune_hparams
from network_utils import get_dataset, create_loss, add_summary, restore_map
from model import model, model_arg_scope

import tensorflow as tf
import tensorflow.contrib.slim as slim

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='training network')

parser.add_argument('--master', default='', type=str, help='BNS name of the TensorFlow master to use')
parser.add_argument('--task_id', default=0, type=int, help='The Task ID. This value is used '
                                                           'when training with multiple workers to '
                                                           'identify each worker.')
parser.add_argument('--train_log_dir', default='logs/', type=str, help='Directory where to write event logs.')
parser.add_argument('--save_summaries_steps', default=120, type=int, help='The frequency with which'
                                                                          ' summaries are saved, in seconds.')
parser.add_argument('--save_interval_secs', default=300, type=int, help='The frequency with which '
                                                                        'the model is saved, in seconds.')
parser.add_argument('--print_loss_steps', default=100, type=int, help='The frequency with which '
                                                                      'the losses are printed, in steps.')
parser.add_argument('--source_dir', default='', type=str, help='The directory where the source datasets can be found.')
parser.add_argument('--target_dir', default='', type=str, help='The directory where the target datasets can be found.')
parser.add_argument('--num_readers', default=2, type=int, help='The number of parallel readers '
                                                               'that read data from the dataset.')
parser.add_argument('--num_steps', default=20000, type=int, help='The max number of gradient steps to take '
                                                                 'during training.')
parser.add_argument('--num_preprocessing_threads', default=2, type=int, help='The number of threads '
                                                                             'used to create the batches.')
parser.add_argument('--hparams', default='', type=str, help='Comma separated hyper parameter values')
parser.add_argument('--from_adapt_checkpoint', default=False, type=bool, help='Whether load checkpoint '
                                                                              'from graspnet checkpoint '
                                                                              'or classification checkpoint.')
parser.add_argument('--checkpoint_dir', default='', type=str, help='The directory where the checkpoint can be found')
args = parser.parse_args()
num_classes = 18


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = create_finetune_hparams()
    for path in [args.train_log_dir]:
        if not tf.gfile.Exists(path):
            tf.gfile.MakeDirs(path)
    hparams_filename = os.path.join(args.train_log_dir, 'hparams.json')
    with tf.gfile.FastGFile(hparams_filename, 'w') as f:
        f.write(hparams.to_json())
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(args.task_id)):
            global_step = tf.train.get_or_create_global_step()
            images_p, class_labels_p, theta_labels_p = get_dataset(os.path.join(args.target_dir, 'positive'),
                                                                   args.num_readers,
                                                                   args.num_preprocessing_threads,
                                                                   hparams)
            images_n, class_labels_n, theta_labels_n = get_dataset(os.path.join(args.target_dir, 'negative'),
                                                                   args.num_readers,
                                                                   args.num_preprocessing_threads,
                                                                   hparams)
            images = tf.concat([images_p, images_n], axis=0)
            class_labels = tf.concat([class_labels_p, class_labels_n], axis=0)
            theta_labels = tf.concat([theta_labels_p, theta_labels_n], axis=0)
            with slim.arg_scope(model_arg_scope()):
                net, end_points = model(inputs=images,
                                        num_classes=num_classes,
                                        is_training=True,
                                        dropout_keep_prob=0.7,
                                        scope=hparams.scope)
            loss, accuracy = create_loss(net,
                                         end_points,
                                         class_labels,
                                         theta_labels)
            learning_rate = hparams.learning_rate
            if hparams.lr_decay_step:
                learning_rate = tf.train.exponential_decay(hparams.learning_rate,
                                                           tf.train.get_or_create_global_step(),
                                                           decay_steps=hparams.lr_decay_step,
                                                           decay_rate=hparams.lr_decay_rate,
                                                           staircase=True)
            tf.summary.scalar('Learning_rate', learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = slim.learning.create_train_op(loss, optimizer)
            add_summary(images, end_points, loss, accuracy, scope=hparams.scope)
            summary_op = tf.summary.merge_all()
            variable_map = restore_map(from_adapt_checkpoint=args.from_adapt_checkpoint,
                                       scope=hparams.scope,
                                       model_name='source_only',
                                       checkpoint_exclude_scopes=['adapt', 'fc8'])
            init_saver = tf.train.Saver(variable_map)

            def initializer_fn(sess):
                init_saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
                tf.logging.info('Successfully load pretrained checkpoint.')

            init_fn = initializer_fn
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False)
            session_config.gpu_options.allow_growth = True
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=args.save_interval_secs,
                                   max_to_keep=100)

            slim.learning.train(train_op,
                                logdir=args.train_log_dir,
                                master=args.master,
                                global_step=global_step,
                                session_config=session_config,
                                init_fn=init_fn,
                                summary_op=summary_op,
                                number_of_steps=args.num_steps,
                                startup_delay_steps=15,
                                save_summaries_secs=args.save_summaries_steps,
                                saver=saver)


if __name__ == '__main__':
    main()

