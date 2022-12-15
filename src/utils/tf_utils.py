import tensorflow as tf
import numpy as np


def tensor_spec_for_batch(dummy_batch):
    """Returns a TensorSpec for the given batch."""
    if hasattr(dummy_batch, '_asdict'):
        dummy_batch = dummy_batch._asdict()

    def _get_tensor_spec(tensor):
        # Convert input to tensors, possibly from nested lists that need to be
        # converted to a single top-level tensor.
        tensor = tf.convert_to_tensor(tensor)
        # Remove the batch dimension and leave it unspecified.
        spec = tf.TensorSpec(
            shape=[None] + tensor.shape.dims[1:], dtype=tensor.dtype)
        return spec

    return tf.nest.map_structure(_get_tensor_spec, dummy_batch)


def vars_to_type(var_struct):
    return tf.nest.map_structure(lambda v: tf.TensorSpec.from_tensor(v), var_struct)


def get_weights(model):
    """Returns tensors of model weights, in the order of the variables."""
    return [v.read_value() for v in model.weights]


def sample_clients(client_list, sampling_prob):
    x = np.random.uniform(size=len(client_list))
    sampled_clients = [client_list[i] for i in range(len(client_list)) if x[i] < sampling_prob]
    if len(sampled_clients) == 0:
        sampled_clients = list(np.random.choice(client_list, 1))

    return sampled_clients


