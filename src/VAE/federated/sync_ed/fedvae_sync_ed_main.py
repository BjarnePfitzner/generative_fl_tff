import tensorflow as tf

from src.VAE.federated.sync_ed.fedvae_sync_ed_tf import ClientState


def get_initial_client_states(client_ids, max_local_steps, model, cfg):
    return {client_id: ClientState(client_index=tf.constant(i),
                                   iters_count=tf.constant(0),
                                   max_iters=tf.constant(max_local_steps[i]))
            for i, client_id in enumerate(client_ids)}


def set_eval_model_weights(eval_model, weights):
    eval_model.set_weights(weights)
