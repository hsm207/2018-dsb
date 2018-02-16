from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU


class ConvBlock:
    def __init__(self, filters, kernel_size, data_format='channels_first', padding='same', strides=1, alpha=0.1,
                 is_final=False):
        """
        A layer that performs a the following sequence of operations:
            1. 2D Convolution with no activation function
            2. Batchnorm
            3. Leaky ReLu activation function

        If is_final is set to True, then layer will perform the following sequence of operations:
            1. Batchnorm
            2. 2D Convolution with no activation function
            3. Sigmoid activation function

        :param filters: The number of filters for the 2D convolutional layer
        :param kernel_size: The kernel size for the 2D convolutional layer
        :param data_format: The data format of the input to the 2D convolutional layer
        :param padding: The padding to use for the convolutional layer
        :param strides: The strides to use for the convolutional layer
        :param alpha: The parameter of the leaky ReLu activation
        :param is_final: Boolean flag to signal if this block is an intermediary or final block
        """
        channel_axis = 1 if data_format == 'channels_first' else -1
        self.is_final = is_final

        self.conv = Conv2D(filters,
                           kernel_size,
                           strides,
                           padding,
                           data_format,
                           activation='linear',
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

        self.bn = BatchNormalization(axis=channel_axis)

        self.activation = LeakyReLU(alpha=alpha) if not is_final else sigmoid

    def _forward_pass_regular(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.activation(x)

        return x

    def _forward_pass_final(self, features):
        x = self.bn(features)
        x = self.conv(x)
        x = self.activation(x)

        return x

    def __call__(self, features):
        if not self.is_final:
            x = self._forward_pass_regular(features)
        else:
            x = self._forward_pass_final(features)

        return x
