import tensorflow as tf

from src.utils.optimizer_utils import get_optimizer_state, initialize_optimizer_vars
from src.VAE.losses.abstract_vae_loss import AbstractVaeLossFns


def create_vae_train_fn(
        vae_loss_fns: AbstractVaeLossFns,
        encoder_optimizer: tf.keras.optimizers.Optimizer,
        decoder_optimizer: tf.keras.optimizers.Optimizer,
        local_dp=False):
    if encoder_optimizer.variables():
        raise ValueError(
            'Expected encoder_optimizer to not have been used previously, but '
            'variables were already initialized.')
    if decoder_optimizer.variables():
        raise ValueError(
            'Expected decoder_optimizer to not have been used previously, but '
            'variables were already initialized.')
    if local_dp:
        reduce = 'none'
    else:
        reduce = 'mean'

    def vae_train_fn(model: tf.keras.Model,
                     model_inputs,
                     labels,
                     global_round,
                     new_encoder_optimizer_state=None):
        """Trains the model on a single batch.
        Args:
          model: The VAE model.
          model_inputs: A batch of inputs (usually images) for the VAE.
          labels: A batch of labels corresponding to the inputs.
          global_round: The current glob al FL round for beta calculation
          new_optimizer_states: Possible optimizer states to overwrite the current ones with.
        Returns:
          The number of examples trained on.
        """

        def encoder_loss():
            """Does the forward pass and computes losses for the generator."""
            # N.B. The complete pass must be inside loss() for gradient tracing.
            return vae_loss_fns.encoder_loss(model, model_inputs, labels, global_round, reduce=reduce)

        def decoder_loss():
            """Does the forward pass and computes losses for the generator."""
            # N.B. The complete pass must be inside loss() for gradient tracing.
            return vae_loss_fns.decoder_loss(model, model_inputs, labels, global_round, reduce=reduce)

        encoder_optimizer_state = get_optimizer_state(encoder_optimizer)
        if new_encoder_optimizer_state is not None:
            # if optimizer is uninitialised, initialise vars
            try:
                tf.nest.assert_same_structure(encoder_optimizer_state, new_encoder_optimizer_state)
            except ValueError:
                initialize_optimizer_vars(encoder_optimizer, model.encoder)
                encoder_optimizer_state = get_optimizer_state(encoder_optimizer)
                tf.nest.assert_same_structure(encoder_optimizer_state, new_encoder_optimizer_state)
            tf.nest.map_structure(lambda a, b: a.assign(b), encoder_optimizer_state, new_encoder_optimizer_state)

        dec_tape = tf.GradientTape()
        with tf.GradientTape() as enc_tape, dec_tape:
            # Encoder Loss can hold complete VAE loss for KL_Loss
            enc_loss_dict = encoder_loss()

        if local_dp:
            e_grads_and_vars = encoder_optimizer._compute_gradients(enc_loss_dict['encoder_loss'],
                                                                    model.encoder.trainable_variables,
                                                                    tape=enc_tape)
        else:
            e_grads = enc_tape.gradient(enc_loss_dict['encoder_loss'], model.encoder.trainable_variables)
            e_grads_and_vars = zip(e_grads, model.encoder.trainable_variables)
        encoder_optimizer.apply_gradients(e_grads_and_vars)

        with dec_tape:
            dec_loss_dict = decoder_loss()
        if len(dec_loss_dict) == 0:
            # For KL loss
            dec_loss_dict['decoder_loss'] = enc_loss_dict['decoder_loss']
        if local_dp:
            d_grads_and_vars = decoder_optimizer._compute_gradients(dec_loss_dict['decoder_loss'],
                                                                    model.decoder.trainable_variables,
                                                                    tape=dec_tape)
        else:
            d_grads = dec_tape.gradient(dec_loss_dict['decoder_loss'], model.decoder.trainable_variables)
            d_grads_and_vars = zip(d_grads, model.decoder.trainable_variables)
        decoder_optimizer.apply_gradients(d_grads_and_vars)

        loss_dict = {**enc_loss_dict, **dec_loss_dict}  # for python >= 3.9.0 change to "x | y"
        for key, value in loss_dict.items():
            loss_dict[key] = tf.reduce_mean(value)
        if 'loss' not in loss_dict.keys():
            loss_dict['loss'] = loss_dict['encoder_loss'] + loss_dict['decoder_loss']

        return tf.shape(model_inputs)[0], loss_dict, encoder_optimizer_state

    return vae_train_fn
