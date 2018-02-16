"""
This module contains classes that implement various model architectures for both the generator and discriminator
"""
import tensorflow as tf

from utils.layers import ConvBlock


class ConvAssaf:
    def __init__(self, data_format='channels_first', padding='same'):
        """
        A 5 layer fully CNN based on the "MICROSCOPY CELL SEGMENTATION VIA ADVERSARIAL NEURAL NETWORKS" paper
        by Assaf Arbelle and Tammy Riklin Raviv
        :param data_format: Data format of the input image
        :param padding: The padding to use when performing the convolutions
        """
        self.data_format = data_format
        self.padding = padding

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

        # since the given masks have no edge annotation, the output at the final convolutional layer
        # is 1 instead of 3 as in the paper
        self.conv5 = tf.make_template('conv5', ConvBlock(kernel_size=1,
                                                         filters=1,
                                                         data_format=self.data_format,
                                                         padding=self.padding,
                                                         is_final=True))

    def __call__(self, features):
        self._build_layers()
        images = features['images']

        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x
