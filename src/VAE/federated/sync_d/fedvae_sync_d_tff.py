import tensorflow_federated as tff

from src.VAE.federated import fedvae_tf
from src.VAE.federated.fedvae_tff import VaeFnsAndTypes
from src.VAE.federated.sync_d import fedvae_sync_d_tf
from src.utils.optimizer_utils import initialize_optimizer_vars


def build_server_initial_state_computation(vae: VaeFnsAndTypes):
    @tff.tf_computation
    def server_init_tf():
        model = vae.model_fn().decoder
        aggregation_optimizer = vae.aggregation_optimizer_fn()

        return fedvae_tf.server_initial_state(model, aggregation_optimizer)

    return server_init_tf


def build_client_computation(vae: VaeFnsAndTypes):
    @tff.tf_computation(tff.SequenceType(vae.model_input_type),
                        vae.client_state_type,
                        vae.server_message_type)
    def client_computation(model_input, client_state, server_message):
        return fedvae_sync_d_tf.client_computation(
            client_state=client_state,
            server_message=server_message,
            model_input_ds=model_input,
            model=vae.model_fn(),
            train_vae_fn=vae.train_vae_fn)

    return client_computation


def build_server_computation(vae: VaeFnsAndTypes, server_state_type: tff.Type):
    @tff.tf_computation(server_state_type,
                        vae.server_model_weights_type,
                        vae.aggregation_process.state_type.member,
                        vae.client_state_type.iters_count)
    def server_computation(server_state, weights_delta, new_aggregation_state, total_iters_count):
        model = vae.model_fn()
        aggregation_optimizer = vae.aggregation_optimizer_fn()
        initialize_optimizer_vars(aggregation_optimizer, model.decoder)

        return fedvae_tf.server_computation(
            server_state=server_state,
            model=model.decoder,
            aggregation_optimizer=aggregation_optimizer,
            new_aggregation_state=new_aggregation_state,
            weights_delta=weights_delta,
            total_iters_count=total_iters_count)

    return server_computation
