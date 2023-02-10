import h5py
import numpy as np
from os.path import join


def save_results(model, history, model_dir):
    model.save(join(model_dir, 'model.h5'))
    file = h5py.File(join(model_dir, 'history.h5'), "w")
    for key, value in history.history.items():
        file.create_dataset(key, data=np.array(value))

    file.close()