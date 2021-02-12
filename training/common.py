import tensorflow as tf
import itertools
from typing import Union

import utils


def train_model_single(model: tf.keras.models.Model, train_set: tf.data.Dataset, run_id: str, run_num: Union[str, int],
                       epochs: int = 15, batch_size: int = 256, exp_decay: bool = True,
                       **kwargs) -> tf.keras.models.Model:
    """
    Trains a single keras model.
    :param model: An instance of a keras model
    :param train_set: A generator containing the training data
    :param run_id: Path whose latter part is the id of the current run
    :param run_num: Number of the current run (used to indicate multiple cold restarts)
    :param epochs: Number of epochs
    :param batch_size: The batch size
    :param exp_decay: Indication to use an exponential decay of the learning rate
    :param kwargs: Arguments to be passed to the checkpoint callback
           - warmup: how many epochs to wait before starting to save weights
           - patiance: every how many epochs to start storing weights
           - verbose: print every time weights are stored (doesn't work properly)
    :return: The same instance of the model as the first argument
    """

    steps_per_epoch = len(train_set) // batch_size + 1
    result_file = 'weights/{}__{}/'.format(run_id, run_num) + 'weights_epoch_{epoch:03d}.h5'  # TODO: dynamic :03d

    callbacks = [utils.callbacks.SimpleModelCheckpoint(result_file, **kwargs)]

    if exp_decay:
        def scheduler(epoch, lr):
            return lr * tf.math.exp(-0.1)

        callbacks += [tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)]

    model.fit(train_set, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    return model


def run_training(model_gen, hparams: dict, data: tf.data.Dataset, run_id: str, num_runs: int = 5, debug: bool = False,
                 **kwargs) -> None:
    """
    Function that handles the training of multiple models for use in ensembles.

    :param model_gen: Factory for producing model instances according to hparams
    :param hparams: A dictionary containing the hyperparameters used for creating the model instance
    :param data: A generator containing the training data
    :param run_id: Path whose latter part is the id of the current experiment
    :param num_runs: Number of epochs
    :param debug: Debug mode. Don't actually run training and store weights, but see if it works properly.
    :param kwargs: Various training arguments, including:
           - exp_decay: Indication to use an exponential decay of the learning rate
           - epochs: Number of epochs for each training
           - batch_size: The batch size
           - warmup: how many epochs to wait before starting to save weights
           - patiance: every how many epochs to start storing weights
           - verbose: print every time weights are stored (doesn't work properly)
    """

    if debug:
        model = model_gen(hparams)
        print('Hparams:')
        for k, v in hparams.items():
            print(' - {}: {}'.format(k, v))

        print()
        model.summary()
        print()

        for x, y in data:
            print('Batch shapes:', x.shape, y.shape)
            model.train_on_batch(x, y)
            break
    else:
        for i in range(num_runs):
            print('Running experiment: "{}"\nIteration: {}/{}'.format(run_id.split('/')[-1], i + 1, num_runs))
            model = model_gen(hparams)
            _ = train_model_single(model, data, run_id, run_num=i, **kwargs)
            del model
            tf.keras.backend.clear_session()


def make_runs(hparam_combinations_dict):
    """
    Function that generates all possible combinations from a dictionary.
    For example:
    >>> combs_dict = {'a': [1, 2, 3], 'b': [1, 2], 'c': [1, 2, 3, 4]}
    >>> combs = make_runs(combs_dict)
    >>> list(combs)
    >>> [{'a': 1, 'b': 1, 'c': 1},
    ...  {'a': 1, 'b': 1, 'c': 2},
    ...  {'a': 1, 'b': 1, 'c': 3},
    ...  {'a': 1, 'b': 1, 'c': 4},
    ...  {'a': 1, 'b': 2, 'c': 1},
    ...  {'a': 1, 'b': 2, 'c': 2},
    ...  {'a': 1, 'b': 2, 'c': 3},
    ...  {'a': 1, 'b': 2, 'c': 4},
    ...  {'a': 2, 'b': 1, 'c': 1},
    ...  {'a': 2, 'b': 1, 'c': 2},
    ...  {'a': 2, 'b': 1, 'c': 3},
    ...  {'a': 2, 'b': 1, 'c': 4},
    ...  {'a': 2, 'b': 2, 'c': 1},
    ...  {'a': 2, 'b': 2, 'c': 2},
    ...  {'a': 2, 'b': 2, 'c': 3},
    ...  {'a': 2, 'b': 2, 'c': 4},
    ...  {'a': 3, 'b': 1, 'c': 1},
    ...  {'a': 3, 'b': 1, 'c': 2},
    ...  {'a': 3, 'b': 1, 'c': 3},
    ...  {'a': 3, 'b': 1, 'c': 4},
    ...  {'a': 3, 'b': 2, 'c': 1},
    ...  {'a': 3, 'b': 2, 'c': 2},
    ...  {'a': 3, 'b': 2, 'c': 3},
    ...  {'a': 3, 'b': 2, 'c': 4}]
    :param hparam_combinations_dict: Dictionary with all values for each hyperparameter.
    :return: List with dictionary with all possible combinations.
    """

    names = hparam_combinations_dict.keys()
    combs = itertools.product(*hparam_combinations_dict.values())

    for c in combs:
        yield dict(zip(names, c))


def register_experiment(experiment_name: str, run_name: str, log_dir: str = './logs', debug: bool = False) -> str:
    """
    Generates an ID for a given experiment and stores it in a registry.

    :param experiment_name: Name of the experiment.
    :param run_name: Name of a given run in the experiment (usually a mashup of hyperparameter values).
    :param log_dir: Directory under which to store the registry.
    :param debug: If True, will create the ID but won't store it in the registry.
    :return: The experiment ID.
    """

    run_hash = hash(run_name)
    run_id = experiment_name + '/' + str(run_hash)

    if not debug:
        with open(log_dir + '/registry.txt', 'a') as f:
            f.write('{}: {}\n'.format(run_id, run_name))

    return run_id
