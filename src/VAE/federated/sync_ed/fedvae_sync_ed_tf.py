import attr
import tensorflow as tf

from src.VAE.federated.fedvae_tf import ClientOutput, ServerMessage


@attr.s(eq=False, frozen=True, slots=True)
class ClientState(object):
    """Structure for state on the client.
    Fields:
    -   `client_index`: The client index integer to map the client state back to the database hosting client states in
        the driver file.
    -   `iters_count`: The number of total iterations a client has computed in the total rounds so far.
    -   `max_iters`: The maximum number of iterations a client is allowed to perform for local DP.
    """
    client_index = attr.ib()
    iters_count = attr.ib()
    max_iters = attr.ib()


@tf.function
def client_computation(
        client_state: ClientState,
        server_message: ServerMessage,
        model_input_ds: tf.data.Dataset,
        model: tf.keras.Model,
        train_vae_fn) -> ClientOutput:
    """Performs client local training of `model` on `dataset`.
    Args:
      client_state: A 'ClientState'.
      server_message: A `BroadcastMessage` from server.
      model_input_ds: A 'tf.data.Dataset'.
      model: A `tff.learning.Model`.
      train_vae_fn: The function to train the VAE.
    Returns:
      A 'ClientOutput`.
    """
    model_weights = model.weights
    initial_weights = server_message.model_weights
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights, initial_weights)

    total_batch_size = tf.constant(0, dtype=tf.int32)
    total_loss = tf.constant(0.0, dtype=tf.float32)   # todo possibly make these lists and average them out
    total_kl_real = tf.constant(0.0, dtype=tf.float32)
    total_kl_fake = tf.constant(0.0, dtype=tf.float32)
    total_kl_rec = tf.constant(0.0, dtype=tf.float32)
    total_rc_loss = tf.constant(0.0, dtype=tf.float32)
    total_encoder_loss = tf.constant(0.0, dtype=tf.float32)
    total_decoder_loss = tf.constant(0.0, dtype=tf.float32)

    iters_count = tf.convert_to_tensor(client_state.iters_count)
    for image, label in iter(model_input_ds):
        batch_size, batch_losses, _ = train_vae_fn(model, image, label, server_message.global_round)
        total_batch_size += batch_size
        total_loss += batch_losses['loss'] * tf.cast(batch_size, tf.float32)
        # total_kl_loss += batch_losses['kl_loss'] * tf.cast(batch_size, tf.float32)
        total_kl_real += batch_losses['kl_real'] * tf.cast(batch_size, tf.float32)  # todo deal with different keys for different loss fns
        total_kl_fake += batch_losses['kl_fake'] * tf.cast(batch_size, tf.float32)
        total_kl_rec += batch_losses['kl_rec'] * tf.cast(batch_size, tf.float32)
        total_rc_loss += batch_losses['rc_loss'] * tf.cast(batch_size, tf.float32)
        total_encoder_loss += batch_losses['encoder_loss'] * tf.cast(batch_size, tf.float32)
        total_decoder_loss += batch_losses['decoder_loss'] * tf.cast(batch_size, tf.float32)
        iters_count += 1
        if tf.math.greater(client_state.max_iters, 0) and tf.math.greater_equal(iters_count, client_state.max_iters):
            break

    weights_delta = tf.nest.map_structure(tf.subtract, model_weights, initial_weights)
    client_weight = tf.cast(total_batch_size, tf.float32)

    loss_dict = {
       'loss': total_loss / client_weight,
       'kl_real': total_kl_real / client_weight,
       'kl_fake': total_kl_fake / client_weight,
       'kl_rec': total_kl_rec / client_weight,
       'rc_loss': total_rc_loss / client_weight,
       'encoder_loss': total_encoder_loss / client_weight,
       'decoder_loss': total_decoder_loss / client_weight
    }

    return ClientOutput(weights_delta, client_weight, loss_dict,
                        ClientState(client_index=client_state.client_index,
                                    iters_count=iters_count,
                                    max_iters=client_state.max_iters))
