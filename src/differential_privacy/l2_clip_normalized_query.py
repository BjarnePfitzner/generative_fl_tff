import collections

import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import dp_query


class L2ClipNormalizedQuery(dp_query.SumAggregationDPQuery):
    """Implements DPQuery interface for Gaussian sum queries.

    Accumulates clipped vectors, then adds Gaussian noise to the sum.
    """

    # pylint: disable=invalid-name
    _GlobalState = collections.namedtuple(
        '_GlobalState', ['l2_norm_clip', 'denominator'])

    def __init__(self, l2_norm_clip, denominator):
        """Initializes the GaussianSumQuery.

        Args:
          stddev: The stddev of the noise added to the sum.
        """
        self._l2_norm_clip = l2_norm_clip
        self._denominator = denominator
        self._ledger = None

    def set_ledger(self, ledger):
        self._ledger = ledger

    def make_global_state(self, l2_norm_clip, denominator):
        """Creates a global state from the given parameters."""
        return self._GlobalState(tf.cast(l2_norm_clip, tf.float32),
                                 tf.cast(denominator, tf.float32))

    def initial_global_state(self):
        return self.make_global_state(self._l2_norm_clip, self._denominator)

    def derive_sample_params(self, global_state):
        return global_state.l2_norm_clip

    def preprocess_record_impl(self, params, record):
        """Clips the l2 norm, returning the clipped record and the l2 norm.

        Args:
          params: The parameters for the sample.
          record: The record to be processed.

        Returns:
          A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
            the structure of preprocessed tensors, and l2_norm is the total l2 norm
            before clipping.
        """
        l2_norm_clip = params
        record_as_list = tf.nest.flatten(record)
        clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
        return tf.nest.pack_sequence_as(record, clipped_as_list), norm

    def preprocess_record(self, params, record):
        preprocessed_record, _ = self.preprocess_record_impl(params, record)
        return preprocessed_record

    def get_noised_result(self, sample_state, global_state):
        def normalize(v):
            return tf.truediv(v, global_state.denominator)

        return (tf.nest.map_structure(normalize, sample_state),
                self._GlobalState(global_state.l2_norm_clip, global_state.denominator))
