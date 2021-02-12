from pathlib import Path
import os
import tensorflow as tf


class SimpleModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, result_file, warmup=0, patience=1, verbose=False, **kwargs):
        """
        Callback for storing the weights of a model every [patience] epochs after a warmup period of [warmup] epochs.

        :param result_file: Where the weights will be stored. This needs to have 'epoch' as a format parameter.
                            For example: '/some/path/weights_{epoch:2d}.h5'
        :param warmup: How many epochs to wait before storing weights.
        :param patience: How many epochs to wait among two consecutive weight storage operations.
        :param verbose: Output message to the user (does not work properly)
        :param kwargs: Does nothing! is used to ignore redundant keyword arguments.
        """
        super().__init__()
        self.result_file = result_file
        self.verbose = verbose
        self.patience = patience
        self.warmup = warmup

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.warmup or (epoch > self.warmup and epoch % self.patience == 0):
            p = Path(self.result_file.format(epoch=epoch))

            if self.verbose:
                print('\nSaving weights to:', str(p))

            if not p.parent.is_dir():
                os.makedirs(str(p.parent))

            self.model.save(str(p))
