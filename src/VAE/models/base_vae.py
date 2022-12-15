import tensorflow as tf
from abc import abstractmethod


class BaseVAE(tf.keras.Model):
    def __init__(self, latent_dim, data_dim, data_ch, n_classes, output_activation_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dim = data_dim
        self.data_ch = data_ch
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.output_activation_fn = output_activation_fn

        image_input = tf.keras.layers.Input((self.data_dim, self.data_dim, self.data_ch), name='input_image')
        label_input = tf.keras.layers.Input(shape=(1,), name='input_label')
        latent_input = tf.keras.Input(shape=(self.latent_dim,), name='input_latent')

        self.encoder = self.define_encoder(image_input, label_input)
        self.decoder = self.define_decoder(latent_input, label_input)

    @abstractmethod
    def define_encoder(self, image_input, label_input):
        pass

    @abstractmethod
    def define_decoder(self, latent_input, label_input):
        pass

    @staticmethod
    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def call(self, inputs, training=False, mask=None):
        return self.decode(inputs[0], inputs[1], training=training)

    def encode(self, x, y, training=True):
        return self.encoder([x, y], training=training)

    def decode(self, z, y, training=True):
        return self.decoder([z, y], training=training)

    def full_pass(self, inputs, training=True):
        mean, logvar = self.encode(*inputs, training=training)
        z = BaseVAE.reparameterize(mean, logvar)
        rec = self.decode(z, inputs[1], training=training)
        return mean, logvar, z, rec

    def get_config(self):
        pass
