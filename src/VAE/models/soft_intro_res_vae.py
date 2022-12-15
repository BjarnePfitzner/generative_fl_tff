import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from src.VAE.models.base_vae import BaseVAE
from src.VAE.models._res_block import _ResidualBlock


CHANNELS = {
    28: [64, 128],
    32: [64, 128, 256],
    128: [64, 128, 256, 512, 512],
    256: [64, 128, 256, 512, 512, 512],
    512: [32, 64, 128, 256, 512, 512, 512],
    1024: [16, 32, 64, 128, 256, 512, 512, 512]
}


class SoftIntroResVAE(BaseVAE):
    # following Soft-IntroVAE implementation at
    # https://github.com/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae/train_soft_intro_vae.py
    def __init__(self, latent_dim, data_dim, data_ch, n_classes, output_activation_fn):
        assert data_dim in CHANNELS.keys(), "CXR VAE is not defined for this data dimension"
        self.channels = CHANNELS[data_dim]
        self.conv_output_shape = None
        super().__init__(latent_dim, data_dim, data_ch, n_classes, output_activation_fn)

    def define_encoder(self, image_input, label_input):
        cc = self.channels[0]

        # Block-1
        x = layers.Conv2D(cc, kernel_size=5, strides=1, padding='same', use_bias=False, name='encoder/conv_1')(image_input)
        x = layers.BatchNormalization(name='encoder/bn_1')(x)
        x = layers.LeakyReLU(alpha=0.2, name='encoder/lrelu_1')(x)
        x = layers.AveragePooling2D(pool_size=2, name='encoder/avg_pool_1')(x)

        i = 2
        for ch in self.channels[1:]:
            # Blocks 2 to num_additional_blocks
            x = _ResidualBlock(inc=cc, outc=ch, scale=1.0, name=f'encoder/resblock_{i}')(x)
            x = layers.AveragePooling2D(pool_size=2, name=f'encoder/avg_pool_{i}')(x)
            cc = ch
            i += 1

        # Final Block
        x = _ResidualBlock(inc=cc, outc=cc, scale=1.0, name=f'encoder/resblock_{i}')(x)
        flatten = layers.Flatten()(x)

        label_one_hot = layers.experimental.preprocessing.CategoryEncoding(num_tokens=self.n_classes)(label_input)
        concat = layers.Concatenate()([flatten, label_one_hot])
        mean_output = layers.Dense(self.latent_dim, name='encoder/output_latent_mean')(concat)
        logvar_output = layers.Dense(self.latent_dim, name='encoder/output_latent_logvar')(concat)

        # Save conv output shape for decoder initial shape
        self.conv_output_shape = x.shape[1:]

        return tf.keras.Model(inputs=[image_input, label_input], outputs=[mean_output, logvar_output], name='encoder')

    def define_decoder(self, latent_input, label_input):
        cc = self.channels[-1]

        label_one_hot = layers.experimental.preprocessing.CategoryEncoding(num_tokens=self.n_classes)(label_input)
        concat = layers.Concatenate()([latent_input, label_one_hot])

        x = layers.Dense(np.prod(self.conv_output_shape), activation='relu', name='decoder/dense_1')(concat)
        x = layers.Reshape(self.conv_output_shape, name='decoder/Reshape')(x)

        i = 1
        for ch in self.channels[::-1]:
            x = _ResidualBlock(inc=cc, outc=ch, scale=1.0, name=f'decoder/resblock_{i}')(x)
            x = layers.UpSampling2D(size=2, interpolation='nearest', name=f'decoder/upsample_{i}')(x)
            cc = ch
            i += 1

        x = _ResidualBlock(inc=cc, outc=cc, scale=1.0, name=f'decoder/resblock_{i}')(x)
        decoder_output = layers.Conv2D(self.data_ch, kernel_size=5, strides=1, padding='same',
                                       activation=self.output_activation_fn, name=f'decoder/conv2d_out')(x)

        return tf.keras.Model(inputs=[latent_input, label_input], outputs=decoder_output, name='decoder')


def define_vae(latent_dim, data_dim, data_ch, n_classes, output_activation_fn='linear'):
    return SoftIntroResVAE(latent_dim=latent_dim,
                           data_dim=data_dim,
                           data_ch=data_ch,
                           n_classes=n_classes,
                           output_activation_fn=output_activation_fn)
