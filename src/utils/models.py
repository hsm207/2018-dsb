"""
This module contains classes that implement various model architectures for both the generator and discriminator
"""
import tensorflow as tf
from tensorflow.python.keras.backend import set_learning_phase
from tensorflow.python.keras.layers import Dense, LeakyReLU

from utils.layers import ConvBlock, RibCage, ConvToFcAdapter


class ConvAssaf:
    def __init__(self, data_format='channels_first', padding='same', use_edges=False):
        """
        A 5 layer fully CNN based on the "MICROSCOPY CELL SEGMENTATION VIA ADVERSARIAL NEURAL NETWORKS" paper
        by Assaf Arbelle and Tammy Riklin Raviv
        :param data_format: Data format of the input image
        :param padding: The padding to use when performing the convolutions
        :param use_edges: Boolean flag to determine if model should use the edge information in masks or not
        """
        self.data_format = data_format
        self.padding = padding
        self.use_edges = use_edges

    def _build_layers(self):
        self.conv1 = tf.make_template('conv1', ConvBlock(kernel_size=9,
                                                         filters=16,
                                                         data_format=self.data_format,
                                                         padding=self.padding))

        self.conv2 = tf.make_template('conv2', ConvBlock(kernel_size=7,
                                                         filters=32,
                                                         data_format=self.data_format,
                                                         padding=self.padding))

        self.conv3 = tf.make_template('conv3', ConvBlock(kernel_size=5,
                                                         filters=64,
                                                         data_format=self.data_format,
                                                         padding=self.padding))

        self.conv4 = tf.make_template('conv4', ConvBlock(kernel_size=4,
                                                         filters=64,
                                                         data_format=self.data_format,
                                                         padding=self.padding))

        # if use_edges is true, then the output from the Generator must be 3 channels, where:
        # channel 0: probability of pixel is black i.e. background
        # channel 1: probability pixel is white i.e. the body of a cell
        # channel 2: 2, probability pixel is an edge of a cell
        #
        # if use_edges is false, then the output is a single channel where each value represents
        # the probability of a pixel is a body of a cell
        if self.use_edges:
            self.conv5 = tf.make_template('conv5', ConvBlock(kernel_size=1,
                                                             filters=3,
                                                             data_format=self.data_format,
                                                             padding=self.padding,
                                                             is_final=True,
                                                             use_edges=True))
        else:
            self.conv5 = tf.make_template('conv5', ConvBlock(kernel_size=1,
                                                             filters=1,
                                                             data_format=self.data_format,
                                                             padding=self.padding,
                                                             is_final=True))

    def __call__(self, features, mode=tf.estimator.ModeKeys.TRAIN):
        set_learning_phase(True) if mode == tf.estimator.ModeKeys.TRAIN else set_learning_phase(False)
        self._build_layers()
        images = features['images']

        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class FullyConvRibCage:
    def __init__(self, data_format='channels_first', alpha=0.1):
        """
        A variant of the Ribcage Discriminator described in the "MICROSCOPY CELL SEGMENTATION VIA ADVERSARIAL
        NEURAL NETWORKS" paper by Assaf Arbelle and Tammy Riklin Raviv

        This version uses global average pooling on the output of the last rib cage to make the model work
        on arbitrary sized images

        :param data_format: The data format of the input to the convolution and max pool layers
        :param alpha: The parameter to the leaky ReLu activation function
        """
        self.data_format = data_format
        self.channel_axis = 1 if data_format == 'channels_first' else -1
        self.alpha = alpha

    def _build_layers(self):
        self.rib1 = tf.make_template('rib1', RibCage(kernel_size=3, filters=32, data_format=self.data_format))
        self.rib2 = tf.make_template('rib2', RibCage(kernel_size=3, filters=64, data_format=self.data_format))
        self.rib3 = tf.make_template('rib3', RibCage(kernel_size=3, filters=128, data_format=self.data_format))

        self.relu = LeakyReLU(alpha=self.alpha)
        self.conv_to_fc = tf.make_template('flatten_conv',
                                           ConvToFcAdapter(output=97344, data_format=self.data_format))

        self.dense1 = Dense(64, use_bias=False, activation='linear', name='dense1')
        self.dense2 = Dense(64, use_bias=False, activation='linear', name='dense2')

        # TFGAN expects the output of the discriminator to be the logits for the probability
        # that image is real.
        # So, no need to call sigmoid.
        self.dense3 = Dense(1, use_bias=False, activation='linear', name='dense3')

    def __call__(self, inputs, conditioning_inputs, mode=tf.estimator.ModeKeys.TRAIN):
        set_learning_phase(True) if mode == tf.estimator.ModeKeys.TRAIN else set_learning_phase(False)
        self._build_layers()

        masks = inputs
        images = conditioning_inputs['images']
        concat_images_masks = tf.concat([images, masks], axis=self.channel_axis)

        # pass the inputs through the rib blocks
        left1, right1, center1 = self.rib1(images, masks, concat_images_masks)
        left2, right2, center2 = self.rib2(left1, right1, center1)
        left3, right3, center3 = self.rib3(left2, right2, center2)

        # flatten the output from the rib blocks using global average pooling
        ribs_output = tf.concat([left3, right3, center3], axis=self.channel_axis)
        ribs_output = self.conv_to_fc(ribs_output)

        # pass the results through to the dense layers
        ribs_output = self.dense1(ribs_output)
        ribs_output = self.relu(ribs_output)

        ribs_output = self.dense2(ribs_output)
        ribs_output = self.relu(ribs_output)

        ribs_output = self.dense3(ribs_output)

        return ribs_output
