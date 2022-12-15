import tensorflow as tf
from tensorflow.keras import layers

from src.VAE.models.base_vae import BaseVAE


class CXRVAE(BaseVAE):
    def define_encoder(self, image_input, label_input):
        e = layers.Conv2D(32, 3, strides=2, padding="same", name='encoder/conv2d_1')(image_input)
        e = layers.ReLU()(e)
        e = layers.Conv2D(64, 3, strides=2, padding="same", name='encoder/conv2d_2')(e)
        e = layers.ReLU()(e)
        e = layers.Conv2D(128, 3, strides=2, padding="same", name='encoder/conv2d_3')(e)
        e = layers.ReLU()(e)
        e = layers.Conv2D(256, 3, strides=2, padding="same", name='encoder/conv2d_4')(e)
        e = layers.ReLU()(e)
        e = layers.Flatten(name='encoder/flatten')(e)

        label_one_hot = layers.experimental.preprocessing.CategoryEncoding(num_tokens=self.n_classes)(label_input)
        concat = layers.Concatenate()([e, label_one_hot])

        mean_output = layers.Dense(self.latent_dim, name='encoder/output_latent_mean')(concat)
        logvar_output = layers.Dense(self.latent_dim, name='encoder/output_latent_logvar')(concat)

        return tf.keras.Model(inputs=[image_input, label_input], outputs=[mean_output, logvar_output], name='encoder')

    def define_decoder(self, latent_input, label_input):
        label_one_hot = layers.experimental.preprocessing.CategoryEncoding(num_tokens=self.n_classes)(label_input)
        concat = layers.Concatenate()([latent_input, label_one_hot])

        d = layers.Dense(8 * 8 * 64, activation="relu", name='decoder/dense_1')(concat)
        d = layers.Reshape((8, 8, 64))(d)

        d = layers.Conv2DTranspose(256, 3, strides=2, padding="same", name='decoder/conv2dt_1')(d)
        d = layers.ReLU()(d)
        d = layers.Conv2DTranspose(128, 3, strides=2, padding="same", name='decoder/conv2dt_2')(d)
        d = layers.ReLU()(d)
        d = layers.Conv2DTranspose(64, 3, strides=2, padding="same", name='decoder/conv2dt_3')(d)
        d = layers.ReLU()(d)
        d = layers.Conv2DTranspose(32, 3, strides=2, padding="same", name='decoder/conv2dt_4')(d)
        d = layers.ReLU()(d)

        decoder_output = layers.Conv2DTranspose(self.data_ch, 3, strides=1, padding="same",
                                                activation=self.output_activation_fn, name='decoder/output_image')(d)

        return tf.keras.Model(inputs=[latent_input, label_input], outputs=decoder_output, name='decoder')


def define_vae(latent_dim, data_dim, data_ch, n_classes, output_activation_fn='linear'):
    return CXRVAE(latent_dim=latent_dim,
                  data_dim=data_dim,
                  data_ch=data_ch,
                  n_classes=n_classes,
                  output_activation_fn=output_activation_fn)
