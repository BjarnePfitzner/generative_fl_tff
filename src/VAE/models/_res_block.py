import tensorflow as tf
from tensorflow.keras import layers


class _ResidualBlock(tf.keras.Model):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0, name='resblock'):
        super(_ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = layers.Conv2D(filters=outc, kernel_size=1, strides=1, padding='valid', groups=1,
                                             use_bias=False, name=f'{name}_conv2d_expand')
        else:
            self.conv_expand = None

        self.conv1 = layers.Conv2D(filters=midc, kernel_size=3, strides=1, padding='same', groups=groups, use_bias=False,
                                   name=f'{name}_conv2d_1')
        self.bn1 = layers.BatchNormalization(name=f'{name}_bn_1')
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.conv2 = layers.Conv2D(filters=outc, kernel_size=3, strides=1, padding='same', groups=groups, use_bias=False,
                                   name=f'{name}_conv2d_2')
        self.bn2 = layers.BatchNormalization(name=f'{name}_bn_2')
        self.relu2 = layers.LeakyReLU(alpha=0.2)

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(inputs)
        else:
            identity_data = inputs

        output = self.relu1(self.bn1(self.conv1(inputs)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(tf.add(output, identity_data))
        return output
