import math

import tensorflow_privacy as tfp


def get_dp_query(noise_multiplier, n_clients_per_round, l2_norm_clip, adaptive=True):
    if adaptive:
        numerator_query = tfp.QuantileAdaptiveClipSumQuery(initial_l2_norm_clip=l2_norm_clip,
                                                           noise_multiplier=noise_multiplier,
                                                           target_unclipped_quantile=0.5,
                                                           learning_rate=0.2,
                                                           clipped_count_stddev=n_clients_per_round / 20,
                                                           expected_num_records=n_clients_per_round,
                                                           geometric_update=True)
    else:
        numerator_query = tfp.GaussianSumQuery(l2_norm_clip=l2_norm_clip,
                                               stddev=noise_multiplier * l2_norm_clip)
    return tfp.NormalizedQuery(numerator_query=numerator_query,
                               denominator=n_clients_per_round)


def calculate_dp_from_rdp(alpha, epsilon, delta):
    return epsilon + (math.log(1 / delta) / alpha), delta
