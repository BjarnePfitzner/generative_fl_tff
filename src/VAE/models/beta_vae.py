import tensorflow as tf
from tensorflow.keras import layers

from src.VAE.models.base_vae import BaseVAE


class CCVAE(BaseVAE):
    def define_encoder(self, image_input, label_input):
        x = layers.ZeroPadding2D(padding=2)(image_input)

        x = layers.Conv2D(32, 4, strides=2, padding="same", name='encoder/conv2d_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(32, 4, strides=2, padding="same", name='encoder/conv2d_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(64, 4, strides=2, padding="same", name='encoder/conv2d_3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(64, 4, strides=2, padding="same", name='encoder/conv2d_4')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Flatten(name='encoder/flatten')(x)

        label_one_hot = layers.experimental.preprocessing.CategoryEncoding(num_tokens=self.n_classes)(label_input)
        concat = layers.Concatenate(name='encoder/concat')([x, label_one_hot])

        mean_output = layers.Dense(self.latent_dim, name='encoder/output_latent_mean')(concat)
        logvar_output = layers.Dense(self.latent_dim, name='encoder/output_latent_logvar')(concat)

        return tf.keras.Model(inputs=[image_input, label_input], outputs=[mean_output, logvar_output], name='encoder')

    def define_decoder(self, latent_input, label_input):
        label_one_hot = layers.experimental.preprocessing.CategoryEncoding(num_tokens=self.n_classes)(label_input)
        concat = layers.Concatenate(name='decoder/concat')([latent_input, label_one_hot])

        x = layers.Dense(2 * 2 * 64, name='decoder/dense')(concat)
        x = layers.Reshape((2, 2, 64), name='decoder/reshape')(x)

        x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", name='decoder/conv2dt_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", name='decoder/conv2dt_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", name='decoder/conv2dt_3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(1, 4, strides=2, padding="same", name='decoder/conv2dt_4')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Cropping2D(2)(x)
        decoder_output = layers.Activation(self.output_activation_fn, name='decoder/output_image')(x)

        return tf.keras.Model(inputs=[latent_input, label_input], outputs=decoder_output, name='decoder')


def define_vae(latent_dim, data_dim, data_ch, n_classes, output_activation_fn='linear'):
    return CCVAE(latent_dim=latent_dim,
                 data_dim=data_dim,
                 data_ch=data_ch,
                 n_classes=n_classes,
                 output_activation_fn=output_activation_fn)
