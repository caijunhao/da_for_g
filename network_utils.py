import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial
import os


def get_dataset(dataset_dir,
                num_readers,
                num_preprocessing_threads,
                hparams,
                reader=None,
                shuffle=True,
                num_epochs=None,
                is_training=True):
    dataset_dir_list = [os.path.join(dataset_dir, filename)
                        for filename in os.listdir(dataset_dir) if filename.endswith('.tfrecord')]
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
        'image/theta/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(channels=3),
        'class_label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
        'theta_label': slim.tfexample_decoder.Tensor('image/theta/label', shape=[]),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(data_sources=dataset_dir_list,
                                   reader=reader,
                                   decoder=decoder,
                                   num_samples=3,
                                   items_to_descriptions=None)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=num_readers,
                                                              shuffle=shuffle,
                                                              num_epochs=num_epochs,
                                                              common_queue_capacity=20 * hparams.batch_size,
                                                              common_queue_min=10 * hparams.batch_size)
    [image, class_label, theta_label] = provider.get(['image', 'class_label', 'theta_label'])
    image = tf.image.resize_images(image, [hparams.image_size, hparams.image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image -= [123.68, 116.779, 103.939]
    if is_training:
        images, class_labels, theta_labels = tf.train.batch([image, class_label, theta_label],
                                                            batch_size=hparams.batch_size,
                                                            num_threads=num_preprocessing_threads,
                                                            capacity=5*hparams.batch_size)
    else:
        images = tf.expand_dims(image,axis=0)
        class_labels = tf.expand_dims(class_label, axis=0)
        theta_labels = tf.expand_dims(theta_label, axis=0)
    return images, class_labels, theta_labels


def create_loss(scores,
                end_points,
                class_labels,
                theta_labels,
                scope=None,
                source_adapt_scope=None,
                target_adapt_scope=None,
                lamb=0.025):
    theta_lebels_one_hot = tf.one_hot(theta_labels, depth=18, on_value=1.0, off_value=0.0)
    theta_acted = tf.reduce_sum(tf.multiply(scores, theta_lebels_one_hot), axis=1, name='theta_acted')
    sig_op = tf.clip_by_value(slim.nn.sigmoid(theta_acted), 0.001, 0.999, name='clipped_sigmoid')
    sig_loss = - tf.to_float(class_labels) * tf.log(sig_op) - \
               (1 - tf.to_float(class_labels)) * tf.log(1 - sig_op)
    sig_loss = tf.reduce_mean(sig_loss)
    conf = tf.equal(tf.to_int32(tf.greater_equal(sig_op, 0.5)),
                    tf.to_int32(tf.greater_equal(tf.to_float(class_labels), 0.1)))
    accuracy = tf.reduce_mean(tf.to_float(conf))
    mmd_loss = None
    if source_adapt_scope is not None:
        source_adapt = tf.squeeze(end_points[scope+'/'+source_adapt_scope])
        target_adapt = tf.squeeze(end_points[scope+'/'+target_adapt_scope])
        mmd_loss = create_mmd_loss(source_adapt, target_adapt)
    loss = sig_loss if mmd_loss is None else sig_loss + lamb * mmd_loss
    return loss, accuracy


def add_summary(images, end_points, loss, accuracy, scope='graspnet'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('image', images[0:1, :, :, :])
    variable_list = slim.get_model_variables(scope=scope)
    for var in variable_list:
        tf.summary.histogram(var.name, var)


def restore_map(from_adapt_checkpoint, scope, model_name, checkpoint_exclude_scopes=None):
    if not from_adapt_checkpoint:
        variables_to_restore = restore_from_pretrained_checkpoint(scope, model_name, checkpoint_exclude_scopes)
        return variables_to_restore
    else:
        variable_list = slim.get_model_variables(scope)
        variables_to_restore = {var.op.name: var for var in variable_list}
        return variables_to_restore


def restore_from_pretrained_checkpoint(scope, model_name, checkpoint_exclude_scopes):
    variable_list = slim.get_model_variables(scope)
    for checkpoint_exclude_scope in checkpoint_exclude_scopes:
        variable_list = [var for var in variable_list if checkpoint_exclude_scope not in var.op.name]
    variables_to_restore = {}
    for var in variable_list:
        if var.name.startswith(scope):
            var_name = var.op.name.replace(scope, model_name)
            variables_to_restore[var_name] = var
    return variables_to_restore


def create_mmd_loss(source, target):
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    loss_value = maximum_mean_discrepancy(source, target, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value)
    return loss_value


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost
