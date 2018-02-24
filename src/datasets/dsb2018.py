from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.io import imread

import utils.preprocessing as preprocess


def _parse_mask_folder(mask_path, use_edges=False):
    masks = Path(mask_path.decode('utf-8')).glob('*')
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


class DsbDataset:
    def __init__(self, root_dir='../datasets/', stage_name='stage1', data_format='channels_first', use_edges=False):
        """
        A class to load the training and test data into tensorflow using the Dataset API

        Assume the images are stored in folders named 'stageX_train' and 'stageX_test' where X is 1 or 2
        :param root_dir: A string representing the root directory where all the files have been downloaded to
        :param stage_name: A string representing the stage prefix of the train and test folder
        """
        train_path = Path(f'{root_dir}/{stage_name}_train/')
        test_path = Path(f'{root_dir}/{stage_name}_test/')

        self.train_images = list(Path(train_path).glob('*/images/*.png'))
        self.train_masks = list(Path(train_path).glob('*/masks'))
        self.test_images = list(Path(test_path).glob('*/images/*.png'))
        self.data_format = data_format
        self.use_edges = use_edges

    def _set_shape_and_channel_dim(self, img, channel_dim):
        # call this function before manipulating channel axis and batching
        img.set_shape((None, None, channel_dim))
        return img

    def _pair_train_images_with_mask(self):
        imgs = [str(path) for path in self.train_images]
        masks = [str(path) for path in self.train_masks]

        imgs = tf.data.Dataset.from_tensor_slices(imgs) \
            .map(tf.read_file) \
            .map(lambda img: tf.image.decode_image(img, channels=3)) \
            .map(lambda img: tf.image.convert_image_dtype(img, tf.float32)) \
            .map(lambda img: self._set_shape_and_channel_dim(img, 3))

        masks = tf.data.Dataset.from_tensor_slices(masks) \
            .map(lambda f: tf.py_func(_parse_mask_folder, [f, self.use_edges], tf.float32)) \
            .map(lambda mask: self._set_shape_and_channel_dim(mask, 1))

        ds = tf.data.Dataset.zip((imgs, masks)) \
            .batch(1)

        return ds

    def get_train_dataset(self):
        """
        Get a dataset where the masks of a given microscopy image into a single mask image

        :return: A Dataset in the form (microscopy image, mask of microscopy image)
        """
        ds = self._pair_train_images_with_mask()
        return ds

    def get_test_dataset(self):
        """
        Get the images to be tested
        :return: A Dataset of test images to make predictions on
        """
        imgs = [str(path) for path in self.test_images]
        imgs = tf.data.Dataset.from_tensor_slices(imgs) \
            .map(tf.read_file) \
            .map(lambda img: tf.image.decode_image(img, channels=3)) \
            .map(lambda img: tf.image.convert_image_dtype(img, tf.float32)) \
            .map(lambda img: self._set_shape_and_channel_dim(img, 3)) \
            .batch(1)

        return imgs

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

            out = {'images': imgs}, masks

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

            out = {'images': imgs}

            return out

        return _test_input_fn
