from tensorflow.keras import layers, models


def get_plain_model(input_shape, output_shape, config):
    dropout = config['dropout']
    model = models.Sequential([layers.InputLayer(input_shape=input_shape)])
    model.add(layers.AveragePooling1D(pool_size=input_shape[0], strides=1))
    model.add(layers.Flatten())
    for layer_config in config['layers']['dense_layers']:
        model.add(layers.Dense(**layer_config))
        if dropout:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(output_shape))

    return model


def get_conv_model(input_shape, output_shape, config):
    dropout = config['dropout']
    model = models.Sequential([layers.InputLayer(input_shape=input_shape)])
    if config['layers']['spin_average']:
        model.add(layers.AveragePooling1D(pool_size=input_shape[0], strides=1))

    model.add(layers.Reshape(input_shape + (1,))) # (batch, vals, qudits, 1))        
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

    model = models.Sequential([layers.InputLayer(input_shape=input_shape)])
    if config['layers']['spin_average']:
        model.add(layers.AveragePooling1D(pool_size=input_shape[0], strides=1))

    model.add(layers.Permute((2, 1))) # (batch, qudits, vals))
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
