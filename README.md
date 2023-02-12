# Deep Quantum Sensing

Deep learning to make predictions from quantum sensor measurements.

## Set up

```
$ conda env create -f environment.yml
```

Note: Plotting the model arquitecture requieres the installation of graphviz. Otherwise, this installation is not necessary.

## Usage

Define the dataset, model and training configuration using a configuration YAML file and run

```
$ python main.py -c [config_file]
```

An example of a configuration file for each possible type of model is included in the folder `config_files`.