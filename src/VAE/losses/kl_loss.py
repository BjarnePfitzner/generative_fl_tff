import tensorflow as tf

from .abstract_vae_loss import AbstractVaeLossFns, kl_divergence, reconstruction_loss


class VaeKlReconstructionLossFns(AbstractVaeLossFns):
    def __init__(self, beta, beta_warmup, cyclical_schedule=False):
        self._beta = float(beta)
        self._beta_warmup = float(beta_warmup)
        self._cyclical_schedule = cyclical_schedule

    def _get_beta(self, global_round):
        if self._beta_warmup == 0:
            return self._beta
        else:
            if self._cyclical_schedule:
                if (global_round // int(self._beta_warmup)) % 2 == 0:
                    # stay at beta
                    current_beta = self._beta * tf.cast(global_round % int(self._beta_warmup), tf.float32) / self._beta_warmup
                else:
                    current_beta = self._beta
                return current_beta
            else:
                return tf.math.minimum(self._beta, self._beta * tf.cast(global_round, tf.float32) / self._beta_warmup)

    def encoder_loss(self, model, model_input, labels, global_round, reduce='mean'):
        # KL Reconstruction loss
        mean, logvar = model.encode(model_input, labels)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z, labels)

        rec_loss = reconstruction_loss(x_logit, model_input, reduce=reduce)
        kl_loss = kl_divergence(mean, logvar, reduce=reduce)

        loss = rec_loss + self._get_beta(global_round) * kl_loss

        if model.losses:
            loss += tf.add_n(model.losses)

        return {
            'encoder_loss': loss,
            'decoder_loss': loss,
            'loss': loss,
            'kl_rec': kl_loss,
            'kl_real': 0.0,
            'kl_fake': 0.0,
            'rc_loss': rec_loss
        }

    def decoder_loss(self, model, model_input, labels, global_round, reduce='mean'):
        return {}
