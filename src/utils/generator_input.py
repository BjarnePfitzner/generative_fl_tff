import tensorflow as tf


def create_generator_noise_dataset(batch_size, noise_dim):
    """Returns a `tf.data.Dataset` of generator noise inputs."""
    return tf.data.Dataset.from_tensors(0).repeat().map(
        lambda _: tf.random.normal([batch_size, noise_dim]))


def create_generator_labels_dataset(batch_size, n_classes, fixed_label=None):
    """Returns a `tf.data.Dataset` of generator label inputs.
        :param batch_size: The batch size of the dataset.
        :param n_classes: the number of different classes for uniform sampling.
        :param fixed_label: If this is set all outputs will be labeled using this label.
    """
    if fixed_label is not None:
        return tf.data.Dataset.from_tensors(tf.cast(fixed_label, tf.uint8)).repeat()
    
    return tf.data.Dataset.from_tensors(0).repeat().map(
        lambda _: tf.cast(tf.random.uniform([batch_size], minval=0, maxval=n_classes, dtype=tf.int32), tf.uint8))


def create_generator_input_dataset(batch_size, n_classes, latent_dim, fixed_label=None):
    return tf.data.Dataset.zip((create_generator_noise_dataset(batch_size, latent_dim),
                                create_generator_labels_dataset(batch_size, n_classes, fixed_label=fixed_label)))
