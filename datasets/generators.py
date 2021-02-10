import tensorflow as tf
import numpy as np
import argparse
import h5py


def seq2seq_generator(data_path: str, batch_size: int = 256, overlap: int = 6, shuffle: bool = True,
                      augmentation: float = 0, debug: bool = False) -> tf.data.Dataset:
    """
    Factory for building TensorFlow data generators for loading time series data.
    Also supports data augmentation and loading series with overlap for backcast.
    :param data_path: Path of a pickle file that contains two arrays: insample and outsample
    :param batch_size: The batch size
    :param overlap: The length with which x and y will overlap (i.e. if len(insample) == 12 and len(outsample) == 6 and
                    overlap == 5, then len(x) == 12 and len(y) == 11). This is done so that the model is trained for
                    also for backcast.
    :param shuffle: True/False whether or not the data will be shuffled.
    :param augmentation: The percentage of the batch that will be augmented data. E.g. if augmentation == 0.75 and
                         batch_size == 200, then each batch will consist of 50 real series and 150 fake ones.
    :param debug: True/False whether or not to print information about the batches.
    :return: A TensorFlow data generator.
    """
    aug_batch_size = int(batch_size * augmentation)
    real_batch_size = int(batch_size * (1 - augmentation))

    if debug:
        print('---------- Generator ----------')
        print('Augmentation percentage:', augmentation)
        print('Batch size:             ', batch_size)
        print('Real batch size:        ', real_batch_size)
        print('Augmentation batch size:', aug_batch_size)
        print('Max aug num:            ', real_batch_size * (real_batch_size - 1) // 2)
        print('------------------------------')

    def augment(x, y):
        random_ind_1 = tf.random.categorical(tf.math.log([[1.] * real_batch_size]), aug_batch_size)
        random_ind_2 = tf.random.categorical(tf.math.log([[1.] * real_batch_size]), aug_batch_size)

        x_aug = (tf.gather(x, random_ind_1) + tf.gather(x, random_ind_2)) / 2
        y_aug = (tf.gather(y, random_ind_1) + tf.gather(y, random_ind_2)) / 2

        return tf.concat([x, tf.squeeze(x_aug, [0])], axis=0), tf.concat([y, tf.squeeze(y_aug, [0])], axis=0)

    # Load data
    with h5py.File(data_path, 'r') as hf:
        x = np.array(hf.get('X'))
        y = np.array(hf.get('y'))

    # Overlap input with output
    if overlap:
        y = np.c_[x[:, -overlap:], y]

    x = x[..., np.newaxis]
    y = y[..., np.newaxis]

    # Tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(buffer_size=len(x))
    data = data.repeat()
    data = data.batch(batch_size=real_batch_size)
    if augmentation:
        data = data.map(augment)
    data = data.prefetch(buffer_size=1)

    data.__class__ = type(data.__class__.__name__, (data.__class__,), {'__len__': lambda self: len(x)})
    return data


if __name__ == '__main__':

    train_set = 'data/yearly_24_train.h5'
    test_set = 'data/yearly_24_test.h5'

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--overlap', type=int, default=6, help='Length of overlap between input and output. '
                                                                     'Outsample length is overlap + 6.')
    parser.add_argument('-a', '--aug', type=float, default=0., help='Percentage of augmented series in batch')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode: Print lots of diagnostic messages.')

    args = parser.parse_args()

    train_gen = seq2seq_generator(train_set, batch_size=256, overlap=args.overlap, shuffle=True,
                                  augmentation=args.aug, debug=args.debug)
    test_gen = seq2seq_generator(test_set, batch_size=256, overlap=args.overlap, shuffle=True,
                                 augmentation=0, debug=args.debug)

    for x, y in train_gen:
        print('Train set:')
        print(x.shape, y.shape)
        break

    for x, y in test_gen:
        print('Test set:')
        print(x.shape, y.shape)
        break