from hparams import create_semi_supervised_with_unlabeled_target_data_hparams
from network_utils import get_dataset, create_semi_supervised_loss2, add_summary, restore_map
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
parser.add_argument('--save_interval_secs', default=500, type=int, help='The frequency with which '
                                                                        'the model is saved, in seconds.')
parser.add_argument('--print_loss_steps', default=100, type=int, help='The frequency with which '
                                                                      'the losses are printed, in steps.')
parser.add_argument('--source_dir', default='', type=str, help='The directory where the source datasets can be found.')
parser.add_argument('--target_dir', default='', type=str, help='The directory where the target datasets can be found.')
parser.add_argument('--target_without_label_dir', default='', type=str, help='The directory where the target datasets '
                                                                             'can be found.')
parser.add_argument('--num_readers', default=2, type=int, help='The number of parallel readers '
                                                               'that read data from the dataset.')
parser.add_argument('--num_steps', default=50000, type=int, help='The max number of gradient steps to take '
                                                                  'during training.')
parser.add_argument('--num_preprocessing_threads', default=2, type=int, help='The number of threads '
                                                                             'used to create the batches.')
parser.add_argument('--hparams', default='', type=str, help='Comma separated hyper parameter values')
parser.add_argument('--from_adapt_checkpoint', default=False, type=bool, help='Whether load checkpoint '
                                                                              'from adapt checkpoint '
                                                                              'or classification checkpoint.')
parser.add_argument('--checkpoint_dir', default='', type=str, help='The directory where the checkpoint can be found')
args = parser.parse_args()
num_classes = 18


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = create_semi_supervised_with_unlabeled_target_data_hparams()
    for path in [args.train_log_dir]:
        if not tf.gfile.Exists(path):
            tf.gfile.MakeDirs(path)
    hparams_filename = os.path.join(args.train_log_dir, 'hparams.json')
    with tf.gfile.FastGFile(hparams_filename, 'w') as f:
        f.write(hparams.to_json())
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(args.task_id)):
            global_step = tf.train.get_or_create_global_step()

            images_p_s, class_labels_p_s, theta_labels_p_s = get_dataset(os.path.join(args.source_dir, 'positive'),
                                                                         args.num_readers,
                                                                         args.num_preprocessing_threads,
                                                                         hparams)
            images_n_s, class_labels_n_s, theta_labels_n_s = get_dataset(os.path.join(args.source_dir, 'negative'),
                                                                         args.num_readers,
                                                                         args.num_preprocessing_threads,
                                                                         hparams)
            images_s = tf.concat([images_p_s, images_n_s], axis=0)
            class_labels_s = tf.concat([class_labels_p_s, class_labels_n_s], axis=0)
            theta_labels_s = tf.concat([theta_labels_p_s, theta_labels_n_s], axis=0)
            with slim.arg_scope(model_arg_scope()):
                net_s, end_points_s = model(inputs=images_s,
                                            num_classes=num_classes,
                                            is_training=True,
                                            dropout_keep_prob=hparams.dropout_keep_prob,
                                            reuse=tf.AUTO_REUSE,
                                            scope=hparams.scope,
                                            adapt_scope='source_adapt_layer',
                                            adapt_dims=128)

            images_p_t_u, _, _ = get_dataset(os.path.join(args.target_without_label_dir, 'positive'),
                                             args.num_readers,
                                             args.num_preprocessing_threads,
                                             hparams)
            images_n_t_u, _, _ = get_dataset(os.path.join(args.target_without_label_dir, 'negative'),
                                             args.num_readers,
                                             args.num_preprocessing_threads,
                                             hparams)
            images_t_u = tf.concat([images_p_t_u, images_n_t_u], axis=0)
            with slim.arg_scope(model_arg_scope()):
                net_t_u, end_points_t_u = model(inputs=images_t_u,
                                                num_classes=num_classes,
                                                is_training=True,
                                                dropout_keep_prob=hparams.dropout_keep_prob,
                                                reuse=tf.AUTO_REUSE,
                                                scope=hparams.scope,
                                                adapt_scope='target_adapt_layer',
                                                adapt_dims=128)

            net = tf.concat([net_s], axis=0)
            images = tf.concat([images_s], axis=0)
            class_labels = tf.concat([class_labels_s], axis=0)
            theta_labels = tf.concat([theta_labels_s], axis=0)
            end_points = {}
            end_points.update(end_points_s)
            end_points[hparams.scope+'/target_adapt_layer_u'] = end_points_t_u[hparams.scope+'/target_adapt_layer']
            loss, accuracy = create_semi_supervised_loss2(net,
                                                          end_points,
                                                          class_labels,
                                                          theta_labels,
                                                          scope=hparams.scope,
                                                          source_adapt_scope='source_adapt_layer',
                                                          target_adapt_scope='target_adapt_layer')
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
            add_summary(images, end_points, loss, accuracy, scope='domain_adapt')
            summary_op = tf.summary.merge_all()
            variable_map = restore_map(from_adapt_checkpoint=args.from_adapt_checkpoint,
                                       scope=hparams.scope,
                                       model_name='source_only',
                                       checkpoint_exclude_scopes=['adapt_layer', 'fc8'])
            init_saver = tf.train.Saver(variable_map)

            def initializer_fn(sess):
                init_saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
                tf.logging.info('Successfully load pretrained checkpoint.')

            init_fn = initializer_fn
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False)
            session_config.gpu_options.allow_growth = True
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=args.save_interval_secs,
                                   max_to_keep=200)

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

