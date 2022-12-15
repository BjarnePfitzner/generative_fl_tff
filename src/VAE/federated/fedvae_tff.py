import attr
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
from src.VAE.federated import fedvae_tf

from src.utils.tf_utils import tensor_spec_for_batch, vars_to_type, get_weights


@attr.s(eq=False, frozen=False)
class VaeFnsAndTypes(object):
    """A container for functions needed to build TFF VAEs.
    This class holds the "context" needed in order to build a complete TFF
    Federated Computation for VAE training, including functions for building
    the models and optimizers.
    The members generally correspond to arguments of the same names
    passed to the functions of `fedvae_tf.py`.
    """
    # Arguments to __init__
    model_fn = attr.ib()
    aggregation_optimizer_fn = attr.ib()
    client_state_fn = attr.ib()
    train_vae_fn = attr.ib()
    dummy_model_input = attr.ib()
    dp_average_query = attr.ib(type=tfp.DPQuery, default=None)
    sync_decoder_only = attr.ib(type=bool, default=False)

    # The aggregation process. If `train_discriminator_dp_average_query` is
    # specified, this will be used to perform the aggregation steps (clipping,
    # noising) necessary for differential privacy (DP). If
    # `train_discriminator_dp_average_query` (i.e., no DP), this will be a simple
    # stateless mean.
    aggregation_process = attr.ib(init=False, type=tff.templates.AggregationProcess, default=None)

    model_input_type = attr.ib(init=False)
    client_state_type = attr.ib(init=False, type=tff.FederatedType)

    def __attrs_post_init__(self):
        self.model_input_type = tensor_spec_for_batch(self.dummy_model_input)

        _model = self.model_fn()
        _ = _model.full_pass(self.dummy_model_input)
        if not isinstance(_model, tf.keras.models.Model):
            raise TypeError('Expected `tf.keras.models.Model`, found {}.'.format(type(_model)))

        if self.sync_decoder_only:
            # only save decoder weights on the server (i.e. for sync_d)
            _model = _model.decoder
        self.server_model_weights_type = vars_to_type(get_weights(_model))

        self.server_message_type = fedvae_tf.ServerMessage(
            model_weights=self.server_model_weights_type,
            global_round=tff.framework.type_from_tensors(0)
        )

        self.client_model_input_type = tff.type_at_clients(
            tff.SequenceType(self.model_input_type)
        )

        self.client_state_type = tff.framework.type_from_tensors(self.client_state_fn())
        self.federated_client_state_type = tff.type_at_clients(
            self.client_state_type
        )

        # The aggregation process. If `train_discriminator_dp_average_query` is
        # specified, this will be used to perform the aggregation steps (clipping,
        # noising) necessary for differential privacy (DP). If
        # `train_discriminator_dp_average_query` (i.e., no DP), this will be a simple
        # stateless mean.
        # Right now, the logic in this library is effectively "if DP use stateful
        # aggregator, else don't use stateful aggregator". An alternative
        # formulation would be to always use a stateful aggregator, but when not
        # using DP default the aggregator to be a stateless mean, e.g.,
        # https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/framework/optimizer_utils.py#L283.
        if self.dp_average_query is not None:
            self.aggregation_process = tff.aggregators.DifferentiallyPrivateFactory(
                query=self.dp_average_query).create(
                value_type=tff.to_type(self.server_model_weights_type))
        else:
            self.aggregation_process = tff.aggregators.MeanFactory().create(
                value_type=tff.to_type(self.server_model_weights_type),
                weight_type=tff.to_type(tf.float32))


def build_federated_vae_process(vae: VaeFnsAndTypes, impl_class):
    client_computation = impl_class.build_client_computation(vae)

    @tff.federated_computation
    def fed_server_initial_state():
        """Orchestration logic for server model initialization."""
        state = tff.federated_eval(impl_class.build_server_initial_state_computation(vae),
                                   tff.SERVER)
        return tff.federated_zip(fedvae_tf.ServerState(
            model_weights=state.model_weights,
            aggregation_optimizer_state=state.aggregation_optimizer_state,
            counters=state.counters,
            aggregation_state=vae.aggregation_process.initialize()
        ))

    @tff.federated_computation(fed_server_initial_state.type_signature.result, vae.client_model_input_type, vae.federated_client_state_type)
    def run_one_round(server_state, federated_dataset, client_states):
        """Orchestration logic for one round of computation.
        Args:
          server_state: A `stateful_fedavg_tf.ServerState`.
          federated_dataset: A federated `tf.data.Dataset` with placement
            `tff.CLIENTS`.
          client_states: A federated `stateful_fedavg_tf.ClientState`.
        Returns:
          A tuple of updated `ServerState` and `tf.Tensor` of average loss.
        """
        server_message = fedvae_tf.ServerMessage(model_weights=server_state.model_weights,
                                                 global_round=server_state.counters['num_rounds'])
        server_message_at_client = tff.federated_broadcast(server_message)

        client_outputs = tff.federated_map(client_computation, (federated_dataset,
                                                                client_states,
                                                                server_message_at_client))

        # todo: replace with if gan.aggregation_process.is_weighted:
        if tf.math.equal(len(vae.aggregation_process.next.type_signature.parameter), 3):
            # Not using differential privacy.
            aggregation_output = vae.aggregation_process.next(
                server_state.aggregation_state,
                client_outputs.weights_delta,
                client_outputs.client_weight
            )
        else:
            # Note that weight goes unused here if the aggregation is involving
            # Differential Privacy; the underlying AggregationProcess doesn't take the
            # parameter, as it just uniformly weights the clients.
            aggregation_output = vae.aggregation_process.next(
                server_state.aggregation_state,
                client_outputs.weights_delta)
        new_aggregation_state = aggregation_output.state
        averaged_weights_delta = aggregation_output.result

        client_weight = client_outputs.client_weight
        train_metrics = tff.federated_mean(client_outputs.train_metrics, weight=client_weight)
        total_iters_count = tff.federated_sum(client_outputs.client_state.iters_count)

        server_computation = impl_class.build_server_computation(vae,
                                                                 server_state.type_signature.member)

        server_state = tff.federated_map(server_computation, (server_state,
                                                              averaged_weights_delta,
                                                              new_aggregation_state,
                                                              total_iters_count))

        return server_state, train_metrics, client_outputs.client_state, client_outputs.weights_delta, averaged_weights_delta

    return tff.templates.IterativeProcess(initialize_fn=fed_server_initial_state, next_fn=run_one_round)
