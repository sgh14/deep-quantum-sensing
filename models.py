from tensorflow.keras import layers, models


def get_plain_model(input_shape, output_shape, config):
    nmeasures = input_shape[0]
    model = models.Sequential()
    model.add(layers.Reshape(input_shape + (1,), input_shape=input_shape))
    model.add(layers.AveragePooling2D(pool_size=(nmeasures, 1), strides=(1, 1)))
    model.add(layers.Flatten())
    for layer_config in config['layers']['dense_layers']:
        model.add(layers.Dense(**layer_config))

    model.add(layers.Dense(output_shape))

    return model


def get_conv_model(input_shape, output_shape, config):
    model = models.Sequential()
    model.add(layers.Reshape(input_shape + (1,), input_shape=input_shape))
    for layer_config in config['layers']['conv_layers']:
        model.add(layers.Conv2D(**layer_config))

    model.add(layers.Flatten())
    for layer_config in config['layers']['dense_layers']:
        model.add(layers.Dense(**layer_config))

    model.add(layers.Dense(output_shape))

    return model


def get_recurrent_model(input_shape, output_shape, config):
    if config['recurrent_type'] == 'LSTM':
        recurrent_layer = layers.LSTM
    elif config['recurrent_type'] == 'GRU':
        recurrent_layer = layers.GRU
    else:
        recurrent_layer = layers.SimpleRNN

    model = models.Sequential()
    model.add(layers.Permute((2, 1), input_shape=input_shape)) # (batch, qubits, measures))
    for layer_config in config['layers']['recurrent_layers']:
        model.add(layers.Bidirectional(recurrent_layer(**layer_config)))

    for layer_config in config['layers']['dense_layers']:
        model.add(layers.Dense(**layer_config))

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
