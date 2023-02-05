import h5py
import numpy as np
from numpy.random import choice
from tensorflow.keras.utils import Sequence


def read_data(data_file):
    file = h5py.File(data_file, 'r')
    basis = np.array(file['/basis'])
    probabilities = np.array(file['/probabilities'])
    labels = np.array(file['/labels'])

    return basis, probabilities, labels


class Generator(Sequence):
    
    def __init__(self, data_file, nmeasures, batch_size, labels_choice):
        self.basis, self.probabilities, labels = read_data(data_file)
        self.labels = labels[:, labels_choice]
        self._indices = np.arange(self.basis.shape[0])

        self.batch_size = batch_size
        self.num_samples = self.probabilities.shape[0] # nconfigs
        self.nmeasures = nmeasures
        self.nqubits = self.basis.shape[1]
                
        self.input_shape = (self.nmeasures, self.nqubits)
        self.output_shape = self.labels.shape[-1]


    def _measure(self, probs):
        chosen_indices = choice(self._indices, size=self.nmeasures, p=probs)
        measures = self.basis[chosen_indices]

        return measures
    

    def __getitem__(self, index):
        ids = np.arange(index*self.batch_size, (index+1)*self.batch_size)
        X = np.array([self._measure(self.probabilities[i]) for i in ids])
        y = self.labels[ids]

        return X, y
    

    def __len__(self):
        return self.num_samples // self.batch_size