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
    -   `encoder_weights`: The client-specific encoder weights
    -   `encoder_optimizer_state`: The client-specific encoder optimizer state
    """
    client_index = attr.ib()
    iters_count = attr.ib()
    max_iters = attr.ib()
    encoder_weights = attr.ib()
    encoder_optimizer_state = attr.ib()


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
      model_input_ds: A 'tf.data.Dataset'.
      model: A `tff.learning.Model`.
      server_message: A `BroadcastMessage` from server.
      train_vae_fn: The function to train the VAE.
    Returns:
      A 'ClientOutput`.
    """
    encoder_model_weights = model.encoder.weights
    decoder_model_weights = model.decoder.weights
    initial_encoder_weights = client_state.encoder_weights
    initial_decoder_weights = server_message.model_weights
    tf.nest.map_structure(lambda v, t: v.assign(t), encoder_model_weights, initial_encoder_weights)
    tf.nest.map_structure(lambda v, t: v.assign(t), decoder_model_weights, initial_decoder_weights)

    total_batch_size = tf.constant(0, dtype=tf.int32)
    total_loss = tf.constant(0.0, dtype=tf.float32)   # todo possibly make these lists and average them out
    total_kl_real = tf.constant(0.0, dtype=tf.float32)
    total_kl_fake = tf.constant(0.0, dtype=tf.float32)
    total_kl_rec = tf.constant(0.0, dtype=tf.float32)
    total_rc_loss = tf.constant(0.0, dtype=tf.float32)
    total_encoder_loss = tf.constant(0.0, dtype=tf.float32)
    total_decoder_loss = tf.constant(0.0, dtype=tf.float32)

    iters_count = tf.convert_to_tensor(client_state.iters_count)
    enc_optimizer_state = client_state.encoder_optimizer_state
    for image, label in iter(model_input_ds):
        # Store optimizer state in client_state
        batch_size, batch_losses, enc_optimizer_state = train_vae_fn(model, image, label, server_message.global_round,
                                                                     new_encoder_optimizer_state=enc_optimizer_state)
        # Reset optimizer state every time
        # batch_size, batch_losses, _ = train_vae_fn(model, image, label, server_message.global_round)

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

    decoder_weights_delta = tf.nest.map_structure(tf.subtract, decoder_model_weights, initial_decoder_weights)
    client_weight = tf.cast(total_batch_size, tf.float32)

    loss_dict = {
        'loss': total_loss / client_weight,
        'kl_real': total_kl_real / client_weight,
        'kl_fake': total_kl_fake / client_weight,
        'kl_rec': total_kl_rec / client_weight,
        'rc_loss': total_rc_loss / client_weight,
        'encoder_loss': total_encoder_loss / client_weight,
        'decoder_loss': total_decoder_loss / client_weight,
    }

    return ClientOutput(decoder_weights_delta, client_weight, loss_dict,
                        ClientState(client_index=client_state.client_index,
                                    iters_count=iters_count,
                                    max_iters=client_state.max_iters,
                                    encoder_weights=encoder_model_weights,
                                    encoder_optimizer_state=enc_optimizer_state))
