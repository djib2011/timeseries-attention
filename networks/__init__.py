import tensorflow as tf
from networks import sequential


def get(**params) -> tf.keras.models.Model:
    """
    Build and compile the tf.keras model with the desired properties.

    :param params: Parameters describing how to build the model
    :return: the desired tf.keras model
    """

    default = {'input_seq_len': 18, 'output_seq_len': 6, 'layer_size': 64, 'encoder_type': 'bi', 'encoder_layers': 2,
               'decoder_type': 'lstm', 'decoder_layers': 1, 'attention_type': 'mul', 'attention_scale': False,
               'attention_dropout': 0., 'attention_causal': False}

    default.update(params)

    if default['encoder_type'] in ('uni', 'bi') and default['decoder_type'] in ('uni', 'bi'):

        return sequential.build_model(default)
