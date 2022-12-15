import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from src.VAE.models.base_vae import BaseVAE

CHANNELS = {
    28: [256, 512],
    32: [128, 256, 512],
    #64: [128, 256, 512],
    64: [32, 64, 128],
    #128: [64, 128, 256, 512, 512],
    #256: [64, 128, 256, 512, 512, 512],
    #512: [32, 64, 128, 256, 512, 512, 512],
    #1024: [16, 32, 64, 128, 256, 512, 512, 512]
}

class DP2CCVAE(BaseVAE):
    # following ConditionalVAE implementation at
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
    def __init__(self, latent_dim, data_dim, data_ch, n_classes, output_activation_fn):
        assert data_dim in CHANNELS.keys(), "CXR VAE is not defined for this data dimension"
        self.channels = CHANNELS[data_dim]
        self.conv_output_shape = None
        super().__init__(latent_dim, data_dim, data_ch, n_classes, output_activation_fn)

    def define_encoder(self, image_input, label_input):
        label_one_hot = layers.experimental.preprocessing.CategoryEncoding(num_tokens=self.n_classes)(label_input)
        class_embedding = layers.Dense(self.data_dim * self.data_dim)(label_one_hot)
        class_embedding = layers.Reshape((self.data_dim, self.data_dim, 1))(class_embedding)
        label_embedding = layers.Conv2D(self.data_ch, 1, strides=1, padding="same", name='encoder/class_embedding')(image_input)

        x = layers.Concatenate(name='encoder/concat', )([label_embedding, class_embedding])

        for i in range(len(self.channels)):
            x = layers.Conv2D(self.channels[i], 3, strides=2, padding="same", name=f'encoder/conv2d_{i}_{self.channels[i]}')(x)
            x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
            x = layers.LeakyReLU(0.01)(x)

        flatten = layers.Flatten()(x)

        mean_output = layers.Dense(self.latent_dim, name='encoder/output_latent_mean')(flatten)
        logvar_output = layers.Dense(self.latent_dim, name='encoder/output_latent_logvar')(flatten)

        # Save conv output shape for decoder initial shape
        self.conv_output_shape = x.shape[1:]

        return tf.keras.Model(inputs=[image_input, label_input], outputs=[mean_output, logvar_output], name='encoder')

    def define_decoder(self, latent_input, label_input):
        ch = [self.channels[0]] + self.channels
        ch.reverse()

        label_one_hot = layers.experimental.preprocessing.CategoryEncoding(num_tokens=self.n_classes)(label_input)
        concat = layers.Concatenate(name='decoder/concat')([latent_input, label_one_hot])

        x = layers.Dense(np.prod(self.conv_output_shape), activation='relu', name='decoder/dense_2')(concat)
        x = layers.Reshape(self.conv_output_shape, name='decoder/reshape')(x)

        for i in range(len(ch) - 1):
            x = layers.Conv2DTranspose(ch[i+1], 3, strides=2, padding="same", name=f'decoder/conv2dt_{i}_{ch[i+1]}')(x)
            x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
            x = layers.LeakyReLU(0.01)(x)

        decoder_output = layers.Conv2D(self.data_ch, 3, strides=1, padding="same", activation=self.output_activation_fn,
                                       name='decoder/output_image')(x)

        return tf.keras.Model(inputs=[latent_input, label_input], outputs=decoder_output, name='decoder')


def define_vae(latent_dim, data_dim, data_ch, n_classes, output_activation_fn='linear'):
    return DP2CCVAE(latent_dim=latent_dim,
                    data_dim=data_dim,
                    data_ch=data_ch,
                    n_classes=n_classes,
                    output_activation_fn=output_activation_fn)
