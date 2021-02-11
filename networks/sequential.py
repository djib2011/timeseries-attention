import tensorflow as tf


def enc2bi_dec1bi(hparams: dict) -> tf.keras.models.Model:
    """
    Factory for creating a forecaster

    :param hparams: a dictionary containing some hyperparameters that determine how the forecaster will be built
    :return: a tf.keras model
    """

    s = hparams['base_layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s * 2))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True))(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out, name='enc2bi_dec1bi')
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def enc2bi_dec1uni(hparams: dict) -> tf.keras.models.Model:
    """
    Factory for creating a forecaster

    :param hparams: a dictionary containing some hyperparameters that determine how the forecaster will be built
    :return: a tf.keras model
    """

    s = hparams['base_layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s * 2))(x)
    x = tf.keras.layers.LSTM(1, return_sequences=True)(x)
    model = tf.keras.models.Model(inp, x, name='enc2bi_dec1uni')
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def enc2bi_dec1uni_attn(hparams: dict) -> tf.keras.models.Model:
    """
    Factory for creating an attention-based forecaster

    :param hparams: a dictionary containing some hyperparameters that determine how the forecaster will be built
    :return: a tf.keras model
    """

    s = hparams['base_layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    enc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    enc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(enc)

    res = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s * 2))(enc)

    dec = tf.keras.layers.LSTM(s * 2, return_sequences=True)(res)

    att = tf.keras.layers.Attention(use_scale=hparams['attention_scale'],
                                    dropout=hparams['attention_dropout'],
                                    causal=hparams['attention_causal'])([dec, enc])

    out = tf.keras.layers.LSTM(1, return_sequences=True)(att)

    model = tf.keras.models.Model(inp, out, name='enc2bi_dec1uni_attn')
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


def enc2bi_dec1bi_attn(hparams: dict) -> tf.keras.models.Model:
    """
    Factory for creating an attention-based forecaster

    :param hparams: a dictionary containing some hyperparameters that determine how the forecaster will be built
    :return: a tf.keras model
    """

    s = hparams['base_layer_size']
    ratio = hparams['input_seq_length'] // hparams['output_seq_length']

    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    enc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    enc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(enc)

    res = tf.keras.layers.Reshape((hparams['output_seq_length'], ratio * s * 2))(enc)

    dec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(res)

    att = tf.keras.layers.Attention(use_scale=hparams['attention_scale'],
                                    dropout=hparams['attention_dropout'],
                                    causal=hparams['attention_causal'])([dec, enc])

    out = tf.keras.layers.LSTM(1, return_sequences=True)(att)

    model = tf.keras.models.Model(inp, out, name='enc2bi_dec1bi_attn')
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


model_mapping = {'enc2bi_dec1bi': enc2bi_dec1bi,
                 'enc2bi_dec1uni': enc2bi_dec1uni,
                 'enc2bi_dec1uni_attn': enc2bi_dec1uni_attn,
                 'enc2bi_dec1bi_attn': enc2bi_dec1bi_attn}


if __name__ == '__main__':

    hp = {'base_layer_size': 128, 'input_seq_length': 18, 'output_seq_length': 6,
          'attention_scale': False, 'attention_dropout': 0., 'attention_causal': False}

    for name, model_gen in model_mapping.items():
        print(name)
        model = model_gen(hp)
        model.summary()
