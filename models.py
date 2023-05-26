import tensorflow as tf
from tensorflow.keras import layers, models


class FrequencyLayer(layers.Layer):
    def __init__(self, spin, **kwargs):
        super(FrequencyLayer, self).__init__(**kwargs)
        self.spin_vals = list(range(-spin, spin + 1))


    def call(self, inputs):
        counts = []
        for s in self.spin_vals:
            counts.append(tf.reduce_sum(tf.cast(tf.equal(inputs, s), tf.int32), axis=1))

        frequencies = tf.divide(tf.stack(counts, axis=1), inputs.shape[1])

        return frequencies
    

def get_plain_model(input_shape, output_shape, config):
    nmeasures = input_shape[0]
    dropout = config['dropout']
    model = models.Sequential()
    model.add(layers.Reshape(input_shape + (1,), input_shape=input_shape))
    model.add(layers.AveragePooling2D(pool_size=(nmeasures, 1), strides=(1, 1)))
    model.add(layers.Flatten())
    for layer_config in config['layers']['dense_layers']:
        model.add(layers.Dense(**layer_config))
        if dropout:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(output_shape))

    return model


def get_conv_model(input_shape, output_shape, config):
    dropout = config['dropout']
    model = models.Sequential()
    if config['layers']['frequency_layer']:
        spin = config['layers']['frequency_layer']
        model.add(FrequencyLayer(spin=spin, input_shape=input_shape))
        input_shape = (2*spin + 1, input_shape[1])
    
    model.add(layers.Reshape(input_shape + (1,), input_shape=input_shape)) # (batch, vals, qudits, 1))
    for layer_config in config['layers']['conv_layers']:
        model.add(layers.Conv2D(**layer_config))
        if dropout:
            model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    for layer_config in config['layers']['dense_layers']:
        model.add(layers.Dense(**layer_config))
        if dropout:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(output_shape))

    return model


def get_recurrent_model(input_shape, output_shape, config):
    dropout = config['dropout']
    if config['recurrent_type'] == 'LSTM':
        recurrent_layer = layers.LSTM
    elif config['recurrent_type'] == 'GRU':
        recurrent_layer = layers.GRU
    else:
        recurrent_layer = layers.SimpleRNN

    model = models.Sequential()
    if config['layers']['frequency_layer']:
        spin = config['layers']['frequency_layer']
        model.add(FrequencyLayer(spin=spin, input_shape=input_shape))
        input_shape = (2*spin + 1, input_shape[1])

    model.add(layers.Permute((2, 1), input_shape=input_shape)) # (batch, qudits, vals))
    for layer_config in config['layers']['recurrent_layers']:
        model.add(layers.Bidirectional(recurrent_layer(**layer_config)))
        if dropout:
            model.add(layers.Dropout(dropout))

    for layer_config in config['layers']['dense_layers']:
        model.add(layers.Dense(**layer_config))
        if dropout:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(output_shape))

    return model


def get_model(input_shape, output_shape, config):
    model_type = config['model_type']
    if model_type=='plain':
        model = get_plain_model(input_shape, output_shape, config)
    elif model_type=='conv':
        model = get_conv_model(input_shape, output_shape, config)
    elif model_type=='recurrent':
        model = get_recurrent_model(input_shape, output_shape, config)
    else:
       raise ValueError('Incorrect model type specified.')

    return model
