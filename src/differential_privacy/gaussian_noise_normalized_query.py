import collections
import distutils

import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query


class GaussianNoiseNormalizedQuery(dp_query.SumAggregationDPQuery):
    """Implements DPQuery interface for Gaussian sum queries.

    Accumulates clipped vectors, then adds Gaussian noise to the sum.
    """

    # pylint: disable=invalid-name
    _GlobalState = collections.namedtuple(
        '_GlobalState', ['stddev', 'denominator'])

    def __init__(self, stddev, denominator):
        """Initializes the GaussianSumQuery.

        Args:
          stddev: The stddev of the noise added to the sum.
        """
        self._stddev = stddev
        self._denominator = denominator
        self._ledger = None

    def set_ledger(self, ledger):
        self._ledger = ledger

    def make_global_state(self, stddev, denominator):
        """Creates a global state from the given parameters."""
        return self._GlobalState(tf.cast(stddev, tf.float32),
                                 tf.cast(denominator, tf.float32))

    def initial_global_state(self):
        return self.make_global_state(self._stddev, self._denominator)

    def derive_sample_params(self, global_state):
        return global_state.stddev

    def preprocess_record(self, params, record):
        return record

    def get_noised_result(self, sample_state, global_state):
        """See base class."""
        if distutils.version.LooseVersion(
                tf.__version__) < distutils.version.LooseVersion('2.0.0'):

            def add_noise(v):
                return v + tf.random.normal(
                    tf.shape(input=v), stddev=global_state.stddev, dtype=v.dtype)
        else:
            random_normal = tf.random_normal_initializer(
                stddev=global_state.stddev)

            def add_noise(v):
                return v + tf.cast(random_normal(tf.shape(input=v)), dtype=v.dtype)

        noised_sum = tf.nest.map_structure(add_noise, sample_state)

        def normalize(v):
            return tf.truediv(v, global_state.denominator)

        return (tf.nest.map_structure(normalize, noised_sum),
                self._GlobalState(global_state.stddev, global_state.denominator))
