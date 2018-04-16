import os

import tensorflow as tf
import tensorflow.contrib.gan as tfgan

from datasets.dsb2018 import DsbDataset
from models.pix2pix import PatchGAN, Unet
from utils.submission import Submitter

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to train the model.')
flags.DEFINE_integer('num_training_examples', 10, 'Number of training examples to use for training.',
                     lower_bound=1, upper_bound=670)
flags.DEFINE_string('model_suffix', None, 'String to append to model\'s model_dir and submission file.')
flags.DEFINE_string('gpu_to_use', '0,1,2,3', 'String to set the CUDA_VISIBLE_DEVICES environment variable.')
flags.DEFINE_integer('train_batch_size', 10, 'The batch size to use for training', lower_bound=1)

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_to_use


def main(unused_args):
    N_TRAIN_IMAGES = FLAGS.num_training_examples
    BATCH_SIZE = FLAGS.train_batch_size
    STEPS_PER_EPOCH = N_TRAIN_IMAGES / BATCH_SIZE
    N_EPOCHS = FLAGS.num_epochs
    N_STEPS = STEPS_PER_EPOCH * N_EPOCHS

    config = tf.estimator.RunConfig(save_checkpoints_steps=STEPS_PER_EPOCH * 10,
                                    save_summary_steps=STEPS_PER_EPOCH * 10,
                                    log_step_count_steps=STEPS_PER_EPOCH * 10,
                                    model_dir='../model_dir/pix2pix_{}'.format(FLAGS.model_suffix),
                                    keep_checkpoint_max=1)

    tf.logging.set_verbosity(tf.logging.INFO)

    gen = Unet(data_format='channels_last', use_edges=True, use_skip_connections=True)
    dis = PatchGAN(data_format='channels_last')
    dataset = DsbDataset(use_edges=True, data_format='channels_last', use_pix2pix=True, training_batch_size=BATCH_SIZE)

    train_input_fn = dataset \
        .get_train_input_fn(take=N_TRAIN_IMAGES, repeat=-1)

    gan_estimator = tfgan.estimator.GANEstimator(generator_fn=gen,
                                                 discriminator_fn=dis,
                                                 generator_loss_fn=tfgan.losses.modified_generator_loss,
                                                 discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
                                                 generator_optimizer=tf.train.AdamOptimizer(learning_rate=2 / 10000,
                                                                                            beta1=0.5),
                                                 discriminator_optimizer=tf.train.AdamOptimizer(learning_rate=2 / 10000,
                                                                                                beta1=0.5),
                                                 config=config)

    tf.logging.info(
        'Training pix2pix model for {:.0f} steps using {} images from the training set with batch size of {} images...' \
            .format(N_STEPS, N_TRAIN_IMAGES, BATCH_SIZE))

    gan_estimator.train(train_input_fn, max_steps=N_STEPS)

    # submission
    tf.logging.info('Generating Kaggle submission file...')
    submitter = Submitter(estimator=gan_estimator, dataset=dataset, use_edges=True, data_format='channels_last')
    submitter.generate_submission_file(file_suffix=FLAGS.model_suffix)
    tf.logging.info('Finished runnning script!')


if __name__ == '__main__':
    flags.mark_flag_as_required('model_suffix')
    tf.app.run()
