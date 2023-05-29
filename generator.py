import h5py
import numpy as np
from numpy.random import choice
from tensorflow.keras.utils import Sequence


def read_data(data_file):
    with h5py.File(data_file, 'r') as file:
        basis = np.array(file['/basis'])
        probabilities = np.array(file['/probabilities'])
        labels = np.array(file['/coords_0'])

    return basis, probabilities, labels


class Generator(Sequence):
    def __init__(
        self,
        data_file,
        nmeasures,
        batch_size,
        labels_choice,
        basis_type='computational'
    ):
        basis, self.probabilities, labels = read_data(data_file)
        self.basis = np.eye(basis.shape[0]) if basis_type=='algebraic' else basis
        labels = labels[:, labels_choice]
        labels = labels/10**np.floor(np.log10(np.max(np.abs(labels))))
        self.labels = labels
        self._indices = np.arange(self.basis.shape[0])

        self.batch_size = batch_size
        self.num_samples = self.probabilities.shape[0] # nconfigs
        self.nmeasures = nmeasures
                
        self.input_shape = (self.nmeasures, self.basis.shape[1])
        self.output_shape = self.labels.shape[-1]


    def _measure(self, probs):
        chosen_indices = choice(self._indices, size=self.nmeasures, p=probs)
        measures = self.basis[chosen_indices]

        return measures
    

    def __getitem__(self, index):
        ids = np.arange(index*self.batch_size, (index+1)*self.batch_size)
        x = np.array([self._measure(self.probabilities[i]) for i in ids])
        y = self.labels[ids]

        return x, y


    def __len__(self):
        return self.num_samples // self.batch_size