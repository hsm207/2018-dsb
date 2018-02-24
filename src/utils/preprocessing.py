import numpy as np
import cv2 as cv


def annotate_mask_edges(mask_img, contour_color=2):
    """
    Annotate a mask image to highlight its edges

    :param mask_img: A binary array of floats representing a mask (0: black, 1: white)
    :param contour_color: An integer used to mark the edge
    :return: mask_img with edges marked with contour_color
    """

    def get_coordinates(img, val):
        return list(zip(*np.where(np.isin(img, val))))

    img_coordinates = get_coordinates(mask_img, 1)

    mask_img, contours, _ = cv.findContours(mask_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    img_with_contours = cv.drawContours(mask_img, contours, -1, contour_color, 0, maxLevel=0)

    contour_coordinates = get_coordinates(img_with_contours, contour_color)

    # Todo: Figure out which mask is causing assertion to fail
    # check that all the points in the contour are within the original image
    # assert all(coord in img_coordinates for coord in
    #            contour_coordinates), 'Found a contour whose points are not in the original mask image!'

    return img_with_contours


def make_mask_grayscale(raw_mask):
    """
    Convert mask images to grayscale
    :param raw_mask: A an array representing a mask (0: black, 255: white)
    :return: A binary version of raw_mask (0: black, 1: white)
    """
    _, bin_img = cv.threshold(raw_mask, 0, 1, cv.THRESH_BINARY)
    return bin_img
