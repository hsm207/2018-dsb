# I don't remember which repo I base this oode on :(
import functools

import tensorflow as tf


class _EncoderBlock:
    def __init__(self, filters, kernel_size=4, strides=2, alpha=0.2, data_format='channels_first',
                 is_first_block=False,
                 padding='same'):
        channel_axis = 1 if data_format == 'channels_first' else -1
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                           data_format=data_format, activation='linear',
                                           kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.bn = tf.keras.layers.BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.1,
                                                     gamma_initializer=tf.random_normal_initializer(0, 0.02))
        self.activation = functools.partial(tf.keras.activations.relu, alpha=alpha)

        self.is_first_bloc = is_first_block

    def __call__(self, features):
        x = self.conv(features)
        x = x if self.is_first_bloc else self.bn(x)
        x = self.activation(x)

        return x


class _DecoderBlock:
    def __init__(self, filters, kernel_size=4, strides=2, alpha=0, dropout_rate=0.5, data_format='channels_first'):
        channel_axis = 1 if data_format == 'channels_first' else -1
        self.conv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                                    padding='same',
                                                    data_format=data_format, activation='linear',
                                                    kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))
        self.bn = tf.keras.layers.BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.1,
                                                     gamma_initializer=tf.keras.initializers.RandomNormal(0, 0.02))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.activation = functools.partial(tf.keras.activations.relu, alpha=alpha)

    def __call__(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.activation(x)

        return x


class Unet:
    def __init__(self,
                 encoder_filters=[64, 128, 256, 512, 512, 512, 512, 512],
                 decoder_dropout_rates=[0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
                 data_format='channels_first',
                 use_edges=False,
                 use_skip_connections=False):

        assert len(encoder_filters) == len(
            decoder_dropout_rates), 'encoder_filters and decorder_dropout_rates must have same length!'
        self.channel_axis = 1 if data_format == 'channels_first' else 3
        self.encoder_filters = encoder_filters
        self.decoder_dropout_rates = decoder_dropout_rates
        self.data_format = data_format
        self.output_channels = 3 if use_edges else 1
        self.use_skip_connections = use_skip_connections

    def __call__(self, features, mode=tf.estimator.ModeKeys.TRAIN):
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.keras.backend.set_learning_phase(True)
        else:
            tf.keras.backend.set_learning_phase(False)

        self._build_layers()

        return self._forward(features, mode)

    def _build_layers(self):
        self.encoders = []
        self.decoders = []

        # fill the list of encoders and decoders
        layer = tf.make_template('C1', _EncoderBlock(filters=self.encoder_filters[0], data_format=self.data_format))
        self.encoders.append(layer)
        for i, filter_size in enumerate(self.encoder_filters[1:], 2):
            layer = tf.make_template('C{}'.format(i), _EncoderBlock(filters=filter_size, data_format=self.data_format))
            self.encoders.append(layer)

        decoder_filters = reversed(self.encoder_filters)
        for i, (filter_size, dropout_rate) in enumerate(zip(decoder_filters, self.decoder_dropout_rates), 1):
            layer = tf.make_template('CD{}'.format(i), _DecoderBlock(filters=filter_size, dropout_rate=dropout_rate,
                                                                     data_format=self.data_format))
            self.decoders.append(layer)

        self.final_conv = tf.make_template('final_conv',
                                           tf.keras.layers.Conv2D(filters=self.output_channels, kernel_size=4,
                                                                  strides=1, padding='same',
                                                                  activation=functools.partial(
                                                                      tf.keras.activations.softmax,
                                                                      axis=self.channel_axis),
                                                                  data_format=self.data_format,
                                                                  kernel_initializer=tf.keras.initializers.RandomNormal(
                                                                      0, 0.02))
                                           )

    def _forward(self, features, mode=tf.estimator.ModeKeys.TRAIN):
        # track the outputs for fun
        self.encoder_outputs = []
        self.decoder_outputs = []
        output = features['image']
        ori_height = tf.cast(features['height'], tf.int32)
        ori_width = tf.cast(features['width'], tf.int32)

        # pass the inputs all the way through the encoding layers while storing the intermediary outputs too
        for encoder in self.encoders:
            output = encoder(output)
            self.encoder_outputs.append(output)

        # no need for skip connection for the first decoder layer
        # note: the last encoder and the first decoder is like the input's latent representation
        output = self.decoders[0](output)
        self.decoder_outputs.append(output)

        # pass the output (together with corresponding skip connections) all the way through the decoding
        # layers while storing the intermediary outputs too
        for i, decoder in enumerate(self.decoders[1:], 1):
            if self.use_skip_connections:

                x = tf.concat([output, self.encoder_outputs[-i - 1]],
                              axis=self.channel_axis,
                              name='skip_connection_{}'.format(i))
            else:
                x = output
            output = decoder(x)
            self.decoder_outputs.append(output)

        output = self.final_conv(output)

        # resize image back to original dimensions only when predicting mask...what is eval is the context of TFGAN?
        if mode == tf.estimator.ModeKeys.PREDICT:
            output = tf.image.resize_images(output, size=tf.concat([ori_height, ori_width], axis=0))
            return output
        else:
            return output


class PatchGAN:
    def __init__(self,
                 encoder_filters=[64, 128, 256, 512],
                 data_format='channels_first'):
        self.channel_axis = 1 if data_format == 'channels_first' else 3
        self.encoder_filters = encoder_filters
        self.data_format = data_format

    def _build_layers(self):

        self.encoders = []
        layer = tf.make_template('C1', _EncoderBlock(filters=self.encoder_filters[0], data_format=self.data_format,
                                                     padding='valid'), create_scope_now_=True)
        self.encoders.append(layer)
        for i, filter_size in enumerate(self.encoder_filters[1:-1], 2):
            layer = tf.make_template('C{}'.format(i), _EncoderBlock(filters=filter_size, data_format=self.data_format,
                                                                    padding='valid'))
            self.encoders.append(layer)

        layer = tf.make_template('C{}'.format(len(self.encoder_filters)),
                                 _EncoderBlock(filters=self.encoder_filters[-1], data_format=self.data_format,
                                               padding='valid',
                                               strides=1)
                                 )
        self.encoders.append(layer)

        self.final_conv = tf.make_template('final_conv',
                                           tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1,
                                                                  padding='valid',
                                                                  kernel_initializer=tf.keras.initializers.RandomNormal(
                                                                      0, 0.02),
                                                                  activation='linear',
                                                                  data_format=self.data_format))

    def __call__(self, inputs, conditioning_inputs, mode=tf.estimator.ModeKeys.TRAIN):
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.keras.backend.set_learning_phase(True)
        else:
            tf.keras.backend.set_learning_phase(False)

        self._build_layers()
        # note that mask is the output from generator and images is the conditioning input
        images = conditioning_inputs['image']
        mask = inputs

        output = tf.concat([mask, images], axis=self.channel_axis)

        for encoder in self.encoders:
            output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            output = encoder(output)

        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = self.final_conv(output)

        return output
