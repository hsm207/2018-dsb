from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import label


class Submitter:

    def __init__(self, estimator, dataset, use_edges=False, data_format='channels_first'):
        """
        A class to generate the predictions on the test set to submit to Kaggle

        :param estimator: An Estimator that will make the predictions
        :param dataset: An instance of the DsbDataset class (this class contains routines to generate the test set)
        """
        self.estimator = estimator
        self.dataset = dataset
        self.use_edges = use_edges
        self.channel_axis = 2 if data_format == 'channels_last' else 0

    def _rle_encoding(self, x):
        '''
        x: numpy array of shape (height, width), 1 - mask, 0 - background
        Returns run length as list
        '''

        dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    def _prob_to_rles(self, x, cut_off=0.5):
        if not self.use_edges:
            lab_img = label(x > cut_off)
        else:
            # collapse the 3 channel mask into a 1 channel mask
            x = np.argmax(x, axis=self.channel_axis)

            # note: position 0 is considered background and position 1 or 2 is considered part of the cell
            lab_img = label(x > 0)

        if lab_img.max() < 1:
            lab_img[0, 0] = 1  # ensure at least one prediction per image
        for i in range(1, lab_img.max() + 1):
            yield self._rle_encoding(lab_img == i)

    def _generate_submission_df(self):
        # based on https://www.kaggle.com/kmader/nuclei-overview-to-submission
        test_ids = self.dataset.test_images
        test_input_fn = self.dataset.get_test_input_fn()
        predicted_masks = self.estimator.predict(test_input_fn)

        tmp_df = pd.DataFrame({'masks': list(predicted_masks)})
        tmp_df['rles'] = tmp_df['masks'].map(lambda x: list(self._prob_to_rles(np.squeeze(x))))
        tmp_df['ImageId'] = [id.parts[3] for id in test_ids]

        out_pred_list = []
        for _, c_row in tmp_df.iterrows():
            for c_rle in c_row['rles']:
                out_pred_list += [dict(ImageId=c_row['ImageId'],
                                       EncodedPixels=' '.join(np.array(c_rle).astype(str)))]
        out_pred_df = pd.DataFrame(out_pred_list)

        return out_pred_df

    def generate_submission_file(self, save_dir='../submissions/', file_suffix=''):
        """
        Create the csv file to upload to Kaggle.

        The file is named 'submission_timestamp.csv'

        :param save_dir: A string representing the directory to save the submission file
        :return: None. This function is called for its side effects only.
        """
        save_dir = Path(save_dir)
        fname = "submission_{}_{}.csv".format(datetime.now().strftime('%Y-%m-%d %H%M %p'), file_suffix)
        df = self._generate_submission_df()
        df.to_csv(save_dir / fname, index=False)
