import tensorflow as tf

from .abstract_vae_loss import AbstractVaeLossFns, kl_divergence, reconstruction_loss


class SoftIntroVaeLossFns(AbstractVaeLossFns):
    def __init__(self, beta_rec, beta_kl, beta_neg, gamma_r):
        self._beta_rec = beta_rec
        self._beta_kl = beta_kl
        self._beta_neg = beta_neg
        self._gamma_r = gamma_r

        # Needed for loss calculation
        self._z = None
        self._current_noise = None

    def encoder_loss(self, model, model_input, labels, global_round, reduce='mean'):
        scale = tf.cast(1 / tf.reduce_prod(model_input.shape[1:]), tf.float32)

        real_mean, real_logvar, self._z, rec = model.full_pass((model_input, labels))

        self._current_noise = tf.random.normal(tf.shape(real_mean))
        fake = model.decode(self._current_noise, labels)

        loss_rec = reconstruction_loss(rec, model_input, reduce=reduce)

        lossE_real_kl = kl_divergence(real_mean, real_logvar, reduce=reduce)

        rec_mean, rec_logvar, z_rec, rec_rec = model.full_pass((tf.stop_gradient(rec), labels))
        fake_mean, fake_logvar, z_fake, rec_fake = model.full_pass((tf.stop_gradient(fake), labels))

        kl_rec = kl_divergence(rec_mean, rec_logvar, reduce='none')
        kl_fake = kl_divergence(fake_mean, fake_logvar, reduce='none')

        loss_rec_rec_e = reconstruction_loss(rec_rec, rec, reduce='none')
        while len(loss_rec_rec_e.shape) > 1:
            loss_rec_rec_e = tf.reduce_sum(loss_rec_rec_e, axis=-1)
        loss_rec_fake_e = reconstruction_loss(rec_fake, fake, reduce='none')
        while len(loss_rec_fake_e.shape) > 1:
            loss_rec_fake_e = tf.reduce_sum(loss_rec_fake_e, axis=-1)

        exp_elbo_rec = tf.exp(-2.0 * scale * (self._beta_rec * loss_rec_rec_e + self._beta_neg * kl_rec))
        exp_elbo_fake = tf.exp(-2 * scale * (self._beta_rec * loss_rec_fake_e + self._beta_neg * kl_fake))
        if reduce == 'mean':
            exp_elbo_rec = tf.reduce_mean(exp_elbo_rec)
            exp_elbo_fake = tf.reduce_mean(exp_elbo_fake)

        lossE_fake = 0.25 * (exp_elbo_rec + exp_elbo_fake)
        lossE_real = scale * (self._beta_rec * loss_rec + self._beta_kl * lossE_real_kl)

        lossE = lossE_fake + lossE_real

        return {
            'encoder_loss': lossE,
            'exp_elbo_fake': exp_elbo_fake,
            'exp_elbo_real': exp_elbo_rec,
            'kl_real': lossE_real_kl
        }

    def decoder_loss(self, model, model_input, labels, global_round, reduce='mean'):
        scale = tf.cast(1 / tf.reduce_prod(model_input.shape[1:]), tf.float32)
        fake = model.decode(self._current_noise, labels)
        rec = model.decode(tf.stop_gradient(self._z), labels)
        loss_rec = reconstruction_loss(model_input, rec, reduce=reduce)

        rec_mean, rec_logvar = model.encode(rec, labels)
        z_rec = model.reparameterize(rec_mean, rec_logvar)

        fake_mean, fake_logvar = model.encode(fake, labels)
        z_fake = model.reparameterize(fake_mean, fake_logvar)

        rec_rec = model.decode(tf.stop_gradient(z_rec), labels)
        rec_fake = model.decode(tf.stop_gradient(z_fake), labels)

        loss_rec_rec = reconstruction_loss(rec_rec, tf.stop_gradient(rec), reduce=reduce)
        loss_fake_rec = reconstruction_loss(rec_fake, tf.stop_gradient(fake), reduce=reduce)

        lossD_rec_kl = kl_divergence(rec_mean, rec_logvar, reduce=reduce)
        lossD_fake_kl = kl_divergence(fake_mean, fake_logvar, reduce=reduce)

        lossD = scale * (loss_rec * self._beta_rec + (lossD_rec_kl + lossD_fake_kl) * 0.5 * self._beta_kl +
                         self._gamma_r * 0.5 * self._beta_rec * (loss_rec_rec + loss_fake_rec))

        return {
            'decoder_loss': lossD,
            'kl_fake': lossD_fake_kl,
            'kl_rec': lossD_rec_kl,
            'rc_loss': loss_rec
        }
