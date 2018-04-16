# Introduction
My work on Kaggle's 2018 Data Science Bowl competition.

I trained a variant of the [pix2pix](https://phillipi.github.io/pix2pix/) model to perform image segmentation on cell microscopy images.

# Status

## 17 April 2018
Unable to make submission for stage 2 because:

1. Input pipeline was created under the assumption that all images have 3 channels but it seems some images in the stage 2 set were 
read by `skimage.io.imread` as grayscale (no channel dimension).

2. Stage 2 had 3,091 images which is taking a long time to run inference on my GPU-less laptop (not feasible to identify problematic images
given tight deadlines).

## 12 March 2018
Obtained stage 1 score of around 0.25 after training for 200 epochs.
