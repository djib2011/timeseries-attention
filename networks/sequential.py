import tensorflow as tf


def build_model(hparams: dict) -> tf.keras.models.Model:
    """
    Factory for creating a (possibly attention-based) encoder-decoder forecaster.

    hparams should include:

        - 'input_seq_length': size of the input sequence
        - 'output_seq_length': size of the output sequence
        - 'encoder_type': type of the encoder layers ('bi' or 'uni')
        - 'num_encoder_layers': number of layers in the encoder
        - 'decoder_type': type of the decoder layers ('bi' or 'uni')
        - 'num_decoder_layers': number of layers in the decoder
        - 'attention_type': what type of attention to use ('add', 'mul', 'self' or None)
        - 'attention_scale': choose whether or not to scale the attention scores or not
        - 'attention_dropout': percentage of dropout for the attention layer
        - 'attention_causal': adds a mask such that position i cannot attend to positions j > i.
                              This prevents the flow of information from the future towards the past.

    :param hparams: a dictionary containing some arguments that determine how the forecaster will be built
    :return: a compiled tf.keras model
    """

    s = hparams['layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']
    name = 'enc{}{}_{}{}'.format(hparams['num_encoder_layers'], hparams['encoder_type'],
                                 hparams['num_decoder_layers'], hparams['decoder_type'])

    if hparams['encoder_type'] == 'uni':
        make_enc_layer = lambda x: tf.keras.layers.LSTM(s, return_sequences=True)(x)
    elif hparams['encoder_type'] == 'bi':
        make_enc_layer = lambda x: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s // 2, return_sequences=True))(x)
    else:
        raise ValueError("encoder_type must be either 'uni' or 'bi'")

    if hparams['decoder_type'] == 'uni':
        make_dec_layer = lambda x: tf.keras.layers.LSTM(s, return_sequences=True)(x)
    elif hparams['decoder_type'] == 'bi':
        make_dec_layer = lambda x: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s // 2, return_sequences=True))(x)
    else:
        raise ValueError("decoder_type must be either 'uni' or 'bi'")

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))

    enc = inp
    for _ in range(hparams['num_encoder_layers']):
        enc = make_enc_layer(enc)

    dec = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s))(enc)

    if hparams['attention_type'] == 'mul':
        dec = make_dec_layer(dec)
        dec = tf.keras.layers.Attention(use_scale=hparams['attention_scale'],
                                        dropout=hparams['attention_dropout'],
                                        causal=hparams['attention_causal'])([dec, enc])
        hparams['num_decoder_layers'] -= 1
        name += '_mulattn'

    elif hparams['attention_type'] == 'add':
        dec = make_dec_layer(dec)
        dec = tf.keras.layers.AdditiveAttention(use_scale=hparams['attention_scale'],
                                                dropout=hparams['attention_dropout'],
                                                causal=hparams['attention_causal'])([dec, enc])

        hparams['num_decoder_layers'] -= 1
        name += '_addattn'

    elif hparams['attention_type'] == 'self-br':
        # apply self-attention before reshapre

        dec = tf.keras.layers.Attention(use_scale=hparams['attention_scale'],
                                        dropout=hparams['attention_dropout'],
                                        causal=hparams['attention_causal'])([enc, enc])

        dec = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s))(dec)

    elif hparams['attention_type'] == 'self-ar' or hparams['attention_type'] == 'self':
        # apply self-attention after reshape

        dec = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s))(enc)

        dec = tf.keras.layers.Attention(use_scale=hparams['attention_scale'],
                                        dropout=hparams['attention_dropout'],
                                        causal=hparams['attention_causal'])([dec, dec])

    for _ in range(hparams['num_decoder_layers']):
        dec = make_dec_layer(dec)

    out = tf.keras.layers.LSTM(1, return_sequences=True)(dec)

    model = tf.keras.models.Model(inp, out, name=name)

    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])

    return model


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--layer-size', type=int, default=64, help='Size of the layers in the network.')
    parser.add_argument('--attention-type', type=str, default='mul', help='Which attention type to use.')
    parser.add_argument('--attention-dropout', type=float, default=0., help='Percentage of dropout for attention layer.')
    parser.add_argument('--attention-causal', type=bool, default=False, help='Option to apply causal attention.')
    parser.add_argument('--encoder-type', type=str, default='bi', help='What type of layers to use for encoder.')
    parser.add_argument('--decoder-type', type=str, default='lstm', help='What type of layers to use for decoder.')
    parser.add_argument('--encoder-layers', type=int, default=2, help='Number of layers to use for the encoder.')
    parser.add_argument('--decoder-layers', type=int, default=1, help='Number of layers to use for the decoder.')

    args = parser.parse_args()

    params = args.__dict__
    params.update({'input_seq_length': 18, 'output_seq_length': 6})

    model = build_model(params)

    model.summary()
