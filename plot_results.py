from matplotlib import pyplot as plt
from tensorflow.keras import utils
import numpy as np
from os.path import join


def plot_history(history, log_scale, model_dir):
    h = history.history
    keys = [key for key in h.keys() if not key.startswith('val_')]
    for key in keys:
        y = np.array([h[key], h['val_' + key]])
        y = np.log10(y) if log_scale else y
        fig, ax = plt.subplots()
        ax.plot(y[0], label='Training')
        ax.plot(y[1], label='Validation')
        ax.set_ylabel(key)
        ax.set_xlabel('Epoch')
        ax.legend()
        fig.savefig(join(model_dir, key + '.png'))


def plot_results(model, history, model_dir, config):
    log_scale = config['log_scale'] if config['log_scale'] else False
    plot_history(history, log_scale, model_dir)
    utils.plot_model(
            model,
            to_file=join(model_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True
        )