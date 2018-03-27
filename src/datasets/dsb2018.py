from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.io import imread

import utils.preprocessing as preprocess


def _parse_mask_folder(mask_path, use_edges=False):
    masks = Path(mask_path).glob('*')
    masks = [imread(path, as_grey=True) for path in masks]

    # binarize masks
    masks = [preprocess.make_mask_grayscale(mask) for mask in masks]

    # annotate the edges
    if use_edges:
        masks = [preprocess.annotate_mask_edges(mask, contour_color=2) for mask in masks]

    # combine individual masks into a single image
    masks = np.sum(np.stack(masks, axis=0), axis=0)

    # insert an axis to represent the channel ('channels_last')
    masks = np.expand_dims(masks, -1)
    # convert to int32, otherwise get:
    # tensorflow.python.framework.errors_impl.UnimplementedError: Unsupported numpy type 8
    masks = masks.astype(np.float32)

    return masks


class TfRecordExampleConverter:

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def encode_example(self, image, mask=None):
        img_height, img_width, img_channels = image.shape
        mask_channels = mask.shape[2] if mask is not None else 0
        img_raw = image.tostring()
        mask_raw = mask.tostring() if mask is not None else b''

        features_dict = {
            'image_raw': self._bytes_feature(img_raw),
            'image_height': self._int64_feature(img_height),
            'image_width': self._int64_feature(img_width),
            'image_channels': self._int64_feature(img_channels),
            'mask_channels': self._int64_feature(mask_channels),
            'mask': self._bytes_feature(mask_raw)
        }

        tfrecord_features = tf.train.Features(feature=features_dict)

        tfrecord_example = tf.train.Example(features=tfrecord_features)

        return tfrecord_example

    def decode_example(self, example_proto):
        features_dict = {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'image_height': tf.FixedLenFeature([], tf.int64),
            'image_width': tf.FixedLenFeature([], tf.int64),
            'image_channels': tf.FixedLenFeature([], tf.int64),
            'mask_channels': tf.FixedLenFeature([], tf.int64),
            'mask': tf.FixedLenFeature([], tf.string)
        }

        decoded_example = tf.parse_single_example(example_proto, features_dict)

        return decoded_example


class DsbDataset:
    def __init__(self, root_dir='../datasets/', stage_name='stage1', data_format='channels_first', use_edges=False,
                 use_pix2pix=False, training_batch_size=1):
        """
        A class to load the training and test data into tensorflow using the Dataset API

        Assume the images are stored in folders named 'stageX_train' and 'stageX_test' where X is 1 or 2
        :param root_dir: A string representing the root directory where all the files have been downloaded to
        :param stage_name: A string representing the stage prefix of the train and test folder
        :param data_format: A string representing the image's data format to feed into a model
        :param use_edges: A boolean indicating whether to automatically annotate an image's edges too
        :param use_pix2pix: A boolean indicating whethet pipeline will be fed to the pix2pix model (current
                            implementation assumes input images are 256 x 256)
        """
        train_path = Path('{}/{}_train/'.format(root_dir, stage_name))
        test_path = Path('{}/{}_test/'.format(root_dir, stage_name))

        self.train_images = list(Path(train_path).glob('*/images/*.png'))
        self.train_masks = list(Path(train_path).glob('*/masks'))
        self.test_images = list(Path(test_path).glob('*/images/*.png'))
        self.data_format = data_format
        self.use_edges = use_edges
        self.use_pix2pix = use_pix2pix
        # Note: Setting batch size to 1 can lead to problmes when training for long iterations
        # See https://discuss.pytorch.org/t/nan-when-i-use-batch-normalization-batchnorm1d/322/16
        self.train_batch_size = training_batch_size

        self.tfrecords_train_path = '{}/tfrecords/{}_train.tfrecords'.format(root_dir, stage_name)
        self.tfrecords_test_path = '{}/tfrecords/{}_test.tfrecords'.format(root_dir, stage_name)

    def _create_train_tfrecords(self):
        imgs = [imread(img_path) for img_path in self.train_images]
        masks = [_parse_mask_folder(mask_path, self.use_edges) for mask_path in self.train_masks]

        writer = tf.python_io.TFRecordWriter(self.tfrecords_train_path)
        encoder = TfRecordExampleConverter()

        print('Converting training images to tfrecords...')
        for img, mask in zip(imgs, masks):
            tfrecord_example = encoder.encode_example(img, mask)
            writer.write(tfrecord_example.SerializeToString())
        print('Finished converting training images to tfrecords!')

        writer.close()

    def _create_test_tfrecords(self):
        imgs = [imread(img_path) for img_path in self.test_images]

        writer = tf.python_io.TFRecordWriter(self.tfrecords_test_path)
        encoder = TfRecordExampleConverter()

        print('Converting test images to tfrecords...')
        for img in imgs:
            tfrecord_example = encoder.encode_example(img)
            writer.write(tfrecord_example.SerializeToString())
        print('Finished converting test images to tfrecords!')

        writer.close()

    def _load_train_tfrecords(self):
        def _parse_decoded_example(decoded_example):
            image_raw = tf.decode_raw(decoded_example['image_raw'], tf.uint8)
            mask_raw = tf.decode_raw(decoded_example['mask'], tf.float32)
            h, w, c = decoded_example['image_height'], decoded_example['image_width'], decoded_example['image_channels']
            mask_c = decoded_example['mask_channels']
            image_shape = tf.stack([h, w, c])
            mask_shape = tf.stack([h, w, mask_c])

            image = tf.reshape(image_raw, image_shape)[:, :, 0:3]
            image = self._set_shape_and_channel_dim(image, 3)
            image = tf.cast(image, tf.float32)
            features = {
                'image': image,
                'height': h,
                'width': w,
                'channels': c
            }

            label = tf.reshape(mask_raw, mask_shape)
            label = self._set_shape_and_channel_dim(label, 1)

            return features, label

        if not Path(self.tfrecords_train_path).exists():
            self._create_train_tfrecords()

        decoder = TfRecordExampleConverter()
        ds = tf.data.TFRecordDataset(self.tfrecords_train_path) \
            .map(decoder.decode_example) \
            .map(_parse_decoded_example)

        if self.use_pix2pix:
            ds = ds \
                .map(lambda features, label: (self._resize_imgs(features, label, 256, 256)))

        if self.use_edges:
            ds = ds \
                .map(lambda features, label: (features, preprocess.one_hot_encode_mask(label)))

        ds = ds.batch(self.train_batch_size)

        return ds

    def _load_test_tfrecords(self):
        def _parse_decoded_example(decoded_example):
            image_raw = tf.decode_raw(decoded_example['image_raw'], tf.uint8)
            h, w, c = decoded_example['image_height'], decoded_example['image_width'], decoded_example['image_channels']

            image_shape = tf.stack([h, w, c])
            image = tf.reshape(image_raw, image_shape)[:, :, 0:3]
            image = self._set_shape_and_channel_dim(image, 3)
            image = tf.cast(image, tf.float32)

            features = {
                'image': image,
                'height': h,
                'width': w,
                'channels': c
            }

            return features

        if not Path(self.tfrecords_test_path).exists():
            self._create_test_tfrecords()

        decoder = TfRecordExampleConverter()
        ds = tf.data.TFRecordDataset(self.tfrecords_test_path) \
            .map(decoder.decode_example) \
            .map(_parse_decoded_example)

        if self.use_pix2pix:
            ds = ds \
                .map(lambda features: self._resize_imgs(features, label=None, new_height=256, new_width=256))

        ds = ds.batch(1)

        return ds

    def _set_shape_and_channel_dim(self, img, channel_dim):
        # call this function before manipulating channel axis and batching
        img.set_shape((None, None, channel_dim))
        return img

    def _resize_imgs(self, features, label, new_height=256, new_width=256):
        img = features['image']
        features['image'] = tf.image.resize_images(img, (new_height, new_width))

        if label is not None:
            label = tf.image.resize_images(label, (new_height, new_width))
            return features, label
        else:
            return features

    def get_train_dataset(self):
        """
        Get a dataset where the masks of a given microscopy image into a single mask image

        :return: A Dataset in the form (microscopy image, mask of microscopy image)
        """
        # ds = self._pair_train_images_with_mask()
        ds = self._load_train_tfrecords()
        return ds

    def get_test_dataset(self):
        """
        Get the images to be tested
        :return: A Dataset of test images to make predictions on
        """

        ds = self._load_test_tfrecords()
        return ds

    def get_train_input_fn(self, take=-1, repeat=1):
        """
        A function to load the training set into a model using the Dataset API
        :param take: Number of examples to take from the training set
        :param repeat: How many passes to iterate over the training data
        :return: A 0 zero argument function that returns a tuple ({'images':train_image}, mask)
        """

        def _train_input_fn():
            ds = self.get_train_dataset() \
                .take(take) \
                .repeat(repeat)

            iter = ds.make_one_shot_iterator()
            imgs, masks = iter.get_next()

            out = imgs, masks

            return out

        return _train_input_fn

    def get_test_input_fn(self, take=-1, repeat=1):
        """
        A function to load the test set into a model using the Dataset API
        :param take: Number of examples to take from the test set
        :param repeat: How many passes to iterate over the test data
        :return: A 0 zero argument function that returns a dict {'images':test_image}
        """

        def _test_input_fn():
            ds = self.get_test_dataset() \
                .take(take) \
                .repeat(repeat)

            iter = ds.make_one_shot_iterator()
            imgs = iter.get_next()

            out = imgs

            return out

        return _test_input_fn
