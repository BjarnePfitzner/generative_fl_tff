from abc import ABC

import tensorflow as tf


class AbstractVaeLossFns(ABC):
    def encoder_loss(self, model, model_input, labels, global_round, reduce='mean'):
        raise NotImplementedError

    def decoder_loss(self, model, model_input, labels, global_round, reduce='mean'):
        raise NotImplementedError


def kl_divergence(mean, logvar, reduce='mean'):
    kl = 0.5 * tf.reduce_sum(tf.exp(logvar) + tf.square(mean) - 1. - logvar, axis=-1)
    if reduce == 'mean':
        return tf.reduce_mean(kl, axis=0)
    return kl


def reconstruction_loss(prediction, target, reduce='mean'):
    prediction_flat = tf.reshape(prediction, (tf.shape(prediction)[0], -1))
    target_flat = tf.reshape(target, (tf.shape(prediction)[0], -1))
    error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(target_flat, prediction_flat)
    error = tf.reduce_sum(error, axis=-1)
    if reduce == 'mean':
        return tf.reduce_mean(error)
    return error


# def reconstruction_loss(prediction, target, reduce='mean'):
#     error = tf.keras.layers.Flatten()(prediction - target)
#     error = error ** 2
#     error = tf.reduce_sum(error, axis=-1)
#     if reduce == 'mean':
#         return tf.reduce_mean(error)
#     return error
