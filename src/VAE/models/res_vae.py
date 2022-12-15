import tensorflow as tf
from tensorflow.keras import layers
from src.VAE.models.base_vae import BaseVAE
from src.VAE.models._res_block import _ResidualBlock


CHANNELS = {
    128: [32, 64, 128, 256, 256],
    256: [64, 128, 256, 512, 512, 512]
}


class ResVAE(BaseVAE):
    def __init__(self, latent_dim, data_dim, data_ch, n_classes, output_activation_fn):
        assert data_dim in CHANNELS.keys(), "CXR VAE is not defined for this data dimension"
        super().__init__(latent_dim, data_dim, data_ch, n_classes, output_activation_fn)

        self.channels = CHANNELS[data_dim]

    def define_encoder(self, image_input, label_input):
        l = layers.Embedding(self.n_classes, 10, name='encoder/embedding')(label_input)
        l = layers.Dense(self.data_dim ** 2, name='encoder/dense')(l)
        l = layers.Reshape((self.data_dim, self.data_dim, self.data_ch))(l)

        # merge = layers.Concatenate(name='encoder/merge')([image_input, l])
        merge = layers.Multiply(name='encoder/merge')([image_input, l])

        cc = self.channels[0]

        # Block-1
        x = layers.Conv2D(cc, kernel_size=5, strides=1, padding='same', use_bias=False, name='encoder/conv_1')(merge)
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
        mean_output = layers.Dense(self.latent_dim, name='encoder/output_latent_mean')(flatten)
        logvar_output = layers.Dense(self.latent_dim, name='encoder/output_latent_logvar')(flatten)

        return tf.keras.Model(inputs=[image_input, label_input], outputs=[mean_output, logvar_output], name='encoder')

    def define_decoder(self, latent_input, label_input):
        cc = self.channels[-1]
        x = layers.Dense(cc*4*4, activation='relu', name='decoder/dense_1')(latent_input)
        x = layers.Reshape((4, 4, cc), name='decoder/Reshape')(x)

        l = layers.Embedding(self.n_classes, 10, name='encoder/embedding')(label_input)
        l = layers.Dense(4 * 4, name='decoder/dense')(l)
        l = layers.Reshape((4, 4, 1), name='decoder/reshape_embedding')(l)

        x = layers.Multiply(name='decoder/merge')([x, l])
        #x = layers.Concatenate(name='decoder/merge')([x, l])

        i = 1
        for ch in self.channels[::-1]:
            x = _ResidualBlock(inc=cc, outc=ch, scale=1.0, name=f'decoder/resblock_{i}')(x)
            x = layers.UpSampling2D(size=2, interpolation='nearest', name=f'decoder/upsample_{i}')(x)
            cc = ch
            i += 1

        x = _ResidualBlock(inc=cc, outc=cc, scale=1.0, name=f'decoder/resblock_{i}')(x)
        decoder_output = layers.Conv2D(1, kernel_size=5, strides=1, padding='same',
                                       activation=self.output_activation_fn, name=f'decoder/conv2d_out')(x)

        return tf.keras.Model(inputs=[latent_input, label_input], outputs=decoder_output, name='decoder')


def define_vae(latent_dim, data_dim, data_ch, n_classes, output_activation_fn='linear'):
    return ResVAE(latent_dim=latent_dim,
                  data_dim=data_dim,
                  data_ch=data_ch,
                  n_classes=n_classes,
                  output_activation_fn=output_activation_fn)
