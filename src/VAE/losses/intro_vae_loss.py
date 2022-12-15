import tensorflow as tf

from .abstract_vae_loss import AbstractVaeLossFns, kl_divergence, reconstruction_loss


class IntroVaeLossFns(AbstractVaeLossFns):
    def __init__(self, alpha, alpha_warmup, beta, m):
        self._alpha = alpha
        self._alpha_warmup = alpha_warmup
        self._beta = beta
        self._m = m

        self.rec = None
        self.fake = None

    def _get_alpha(self, global_round):
        if self._alpha_warmup == 0 or global_round > self._alpha_warmup:
            return self._alpha
        else:
            return 0.0

    def encoder_loss(self, model, model_input, labels, global_round):
        # Encoder Loss
        real_mean, real_logvar, z, rec = model.full_pass((model_input, labels))
        fake = model.decode(model.reparameterize(tf.zeros_like(real_mean), tf.ones_like(real_logvar)), labels)
        self.rec, self.fake = rec, fake
        rec_mean, rec_logvar = model.encode(tf.stop_gradient(rec), labels)
        fake_mean, fake_logvar = model.encode(tf.stop_gradient(fake), labels)

        loss_rec = reconstruction_loss(rec, model_input)

        lossE_real_kl = kl_divergence(real_mean, real_logvar)
        lossE_rec_kl = kl_divergence(rec_mean, rec_logvar)
        lossE_fake_kl = kl_divergence(fake_mean, fake_logvar)

        loss_margin_E = lossE_real_kl + self._get_alpha(global_round) * (tf.nn.relu(self._m - lossE_rec_kl) + tf.nn.relu(self._m - lossE_fake_kl))

        lossE = loss_margin_E + self._beta * loss_rec

        return {
            'encoder_loss': lossE,
        }

    def decoder_loss(self, model, model_input, labels, global_round):
        # Decoder Loss
        rec_mean, rec_logvar = model.encode(self.rec, labels)
        fake_mean, fake_logvar = model.encode(self.fake, labels)

        lossD_rec_kl = kl_divergence(rec_mean, rec_logvar)
        lossD_fake_kl = kl_divergence(fake_mean, fake_logvar)

        loss_margin_D = self._get_alpha(global_round) * (lossD_rec_kl + lossD_fake_kl)

        loss_rec = reconstruction_loss(self.rec, model_input)

        lossD = loss_margin_D + self._beta * loss_rec

        #total_kl_loss = lossE_real_kl + lossE_rec_kl + lossE_fake_kl + lossD_rec_kl + lossD_fake_kl
        #m_bound = tf.maximum(lossE_rec_kl, lossE_fake_kl)

        return {
            'decoder_loss': lossD,
            'rc_loss': loss_rec
        }
