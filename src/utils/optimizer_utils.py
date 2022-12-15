import attr
import tensorflow as tf
from tensorflow_privacy import VectorizedDPKerasAdamOptimizer, VectorizedDPKerasSGDOptimizer

from src.utils.tf_utils import get_weights


@attr.s(eq=False)
class OptimizerState(object):
    iterations = attr.ib()
    weights = attr.ib()


def get_optimizer_state(optimizer):
    return OptimizerState(
        iterations=optimizer.iterations,
        # The first weight of an optimizer is reserved for the iterations count,
        # see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#get_weights pylint: disable=line-too-long]
        weights=tuple(optimizer.weights[1:]))


def initialize_optimizer_vars(optimizer: tf.keras.optimizers.Optimizer, model):
    """Ensures variables holding the state of `optimizer` are created."""
    zero_gradient = tf.nest.map_structure(tf.zeros_like, get_weights(model))
    if hasattr(optimizer, '_was_dp_gradients_called'):
        optimizer._was_dp_gradients_called = True
    optimizer.apply_gradients(zip(zero_gradient, model.weights))
    assert optimizer.variables()


def get_optimizer(optimizer_config):
    if optimizer_config.type == 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=optimizer_config.lr,
                                       momentum=optimizer_config.momentum,
                                       nesterov=optimizer_config.nesterov)
    elif optimizer_config.type == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=optimizer_config.lr,
                                        beta_1=optimizer_config.beta_1,
                                        beta_2=optimizer_config.beta_2)


def get_dp_optimizer(optimizer_config, dp_config, batch_size):
    if optimizer_config.type == 'SGD':
        return VectorizedDPKerasSGDOptimizer(learning_rate=optimizer_config.lr,
                                             momentum=optimizer_config.momentum,
                                             nesterov=optimizer_config.nesterov,
                                             l2_norm_clip=dp_config.l2_norm_clip,
                                             noise_multiplier=dp_config.noise_multiplier,
                                             num_microbatches=batch_size)
    elif optimizer_config.type == 'Adam':
        return VectorizedDPKerasAdamOptimizer(learning_rate=optimizer_config.lr,
                                              beta_1=optimizer_config.beta_1,
                                              beta_2=optimizer_config.beta_2,
                                              l2_norm_clip=dp_config.l2_norm_clip,
                                              noise_multiplier=dp_config.noise_multiplier,
                                              num_microbatches=batch_size)
