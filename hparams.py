import tensorflow as tf


def create_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.01,
                                          lr_decay_step=10000,
                                          lr_decay_rate=0.95,
                                          dropout_keep_prob=0.7,
                                          batch_size=64,
                                          image_size=224,
                                          lamb=0.025,
                                          scope='mixed')

    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams


def create_source_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.01,
                                          lr_decay_step=10000,
                                          lr_decay_rate=0.95,
                                          dropout_keep_prob=0.7,
                                          batch_size=64,
                                          image_size=224,
                                          lamb=0.025,
                                          scope='source_only')

    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams


def create_target_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.01,
                                          lr_decay_step=10000,
                                          lr_decay_rate=0.95,
                                          dropout_keep_prob=0.7,
                                          batch_size=64,
                                          image_size=224,
                                          lamb=0.025,
                                          scope='target_only')

    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams


def create_finetune_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.001,
                                          lr_decay_step=10000,
                                          lr_decay_rate=0.95,
                                          dropout_keep_prob=0.7,
                                          batch_size=64,
                                          image_size=224,
                                          lamb=0.025,
                                          scope='finetune')

    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams


def create_mixed_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.01,
                                          lr_decay_step=10000,
                                          lr_decay_rate=0.95,
                                          dropout_keep_prob=0.7,
                                          batch_size=32,
                                          image_size=224,
                                          lamb=0.025,
                                          scope='mixed')

    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams


def create_domain_adapt_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.01,
                                          lr_decay_step=10000,
                                          lr_decay_rate=0.95,
                                          dropout_keep_prob=0.7,
                                          batch_size=32,
                                          image_size=224,
                                          adapt_dims=128,
                                          lamb=0.025,
                                          scope='domain_adapt')

    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams


def create_domain_adapt_se_hparams(hparam_string=None):
    hparams = tf.contrib.training.HParams(learning_rate=0.01,
                                          lr_decay_step=10000,
                                          lr_decay_rate=0.95,
                                          dropout_keep_prob=0.7,
                                          batch_size=32,
                                          image_size=224,
                                          adapt_dims=128,
                                          reduction_ratio=16,
                                          lamb=0.025,
                                          scope='domain_adapt_se')

    if hparam_string:
        tf.logging.info('Parsing command line hparams: %s', hparam_string)
        hparams.parse(hparam_string)

    tf.logging.info('Final parsed hparams: %s', hparams.values())
    return hparams