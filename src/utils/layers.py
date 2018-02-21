import tensorflow as tf
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, GlobalAveragePooling2D


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


class RibBlock:
    def __init__(self, filters, kernel_size, strides=1, alpha=0.1, data_format='channels_first', conv_padding='same',
                 maxpool_padding='valid'):
        """
        A layer that performs the following sequence of operations:

            1. 2D Convolution without an activation function
            2. Batchnorm
            3. Leaky ReLu activation function
            4. 2D max pool with (2, 2) stride and kernel size

        :param filters: Number of filters for the convolution layer
        :param kernel_size: Kernel size for the conovlution layer
        :param strides: Strides for the convolution layer
        :param alpha: The parameter for the leaky ReLu activation function
        :param data_format: The data format of the input to the convolution and max pool layer
        :param conv_padding: The padding used for the convolution layer
        :param maxpool_padding: The padding used for the max pool layer
        """
        channel_axis = 1 if data_format == 'channels_first' else -1
        self.conv = Conv2D(filters, kernel_size, strides, padding=conv_padding, data_format=data_format,
                           activation='linear', use_bias=False, kernel_initializer='glorot_uniform')
        self.bn = BatchNormalization(axis=channel_axis)
        self.relu = LeakyReLU(alpha=alpha)
        self.maxpool = MaxPooling2D(pool_size=2, strides=2, padding=maxpool_padding, data_format=data_format)

    def __call__(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class RibCage:
    def __init__(self, filters, kernel_size, strides=1, alpha=0.1, data_format='channels_first', conv_padding='same',
                 maxpool_padding='valid'):
        """
        A Ribcage layer is a layer that consist of 3 Rib Blocks named left, right, and center

        The input to the left and right Rib Blocks are distinct while the input to the center Rib Block is the
        concatenation of the inputs of the left and right Rib Blocks on the channel axis.

        The output of a Ribcage layer is a tuple of 3 tensors, namely:

            (output from left rib block,
            output from right rib block,
            concatenation of output form left, right and center rib block on the channel axis)

        :param filters: Number of filters for the convolution layer
        :param kernel_size: Kernel size for the conovlution layer
        :param strides: Strides for the convolution layer
        :param alpha: The parameter for the leaky ReLu activation function
        :param data_format: The data format of the input to the convolution and max pool layer
        :param conv_padding: The padding used for the convolution layer
        :param maxpool_padding: The padding used for the max pool layer
        """
        self.channel_axis = 1 if data_format == 'channels_first' else -1

        self.left = tf.make_template('left_rib',
                                     RibBlock(filters, kernel_size, strides, alpha, data_format, conv_padding,
                                              maxpool_padding))
        self.center = tf.make_template('spine',
                                       RibBlock(filters // 2, kernel_size, strides, alpha, data_format, conv_padding,
                                                maxpool_padding))

        self.right = tf.make_template('right_rib',
                                      RibBlock(filters, kernel_size, strides, alpha, data_format, conv_padding,
                                               maxpool_padding))

    def __call__(self, images, masks, concat_images_masks):
        left_output = self.left(images)
        right_output = self.right(masks)
        center_output = tf.concat([left_output,
                                   right_output,
                                   self.center(concat_images_masks)], axis=self.channel_axis)

        return left_output, right_output, center_output


class ConvToFcAdapter:
    def __init__(self, output, data_format='channels_first'):
        """
        A layer to substitute flattening the output of a convolutional layer to connect it to a dense layer.

        This layer enables the convolution to be performed on arbitrarily size images.

        The sequence of operations performed are:

            1. 1x1 convolution with output number of filters and no activation function
            2. Global average pooling

        :param output: The number of neurons this layer should output
        :param data_format: The data format of the input passed to the convolution layer
        """
        self.conv = Conv2D(filters=output, kernel_size=1, activation='linear', data_format=data_format)
        self.global_avg_pool = GlobalAveragePooling2D(data_format=data_format)

    def __call__(self, features):
        x = self.conv(features)
        x = self.global_avg_pool(x)

        return x
