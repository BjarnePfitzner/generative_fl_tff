import tensorflow as tf

from src.VAE.federated.sync_d.fedvae_sync_d_tf import ClientState
from src.utils.optimizer_utils import get_optimizer, get_optimizer_state, initialize_optimizer_vars


def get_initial_client_states(client_ids, max_local_steps, model, cfg):
    vae_model_sample = model.define_vae(cfg.model.latent_dim, cfg.dataset.data_dim, cfg.dataset.data_ch, cfg.dataset.n_classes)
    client_optimizer_sample = get_optimizer(cfg.training.client_optimizer)
    initialize_optimizer_vars(optimizer=client_optimizer_sample, model=vae_model_sample.encoder)

    return {client_id: ClientState(
        client_index=tf.constant(i),
        iters_count=tf.constant(0),
        max_iters=tf.constant(max_local_steps[i]),
        encoder_weights=vae_model_sample.encoder.weights,
        encoder_optimizer_state=get_optimizer_state(client_optimizer_sample)
    ) for i, client_id in enumerate(client_ids)}


def set_eval_model_weights(eval_model, weights):
    eval_model.decoder.set_weights(weights)
