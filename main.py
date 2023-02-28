import argparse
import yaml
from tensorflow.keras import models
import os
import wandb

from generator import Generator
from models import get_model
from save_results import save_results
from plot_results import plot_results


def parse_commandline():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        "-c",
        required=True,
        help="path to YAML configuration file with simulation options")

    return parser.parse_args()


def main():
    args = parse_commandline()
    with open(args.config_file, 'r') as config_file:
        c = yaml.safe_load(config_file)

    wandb.init(
        project="dqsensing",
        config = {
            'Dataset': c['dataset'],
            'Model': c['model'], 
            'Training': c['training']
        }
    )
    config_artifact = wandb.Artifact('config_file', type='config_file')
    config_artifact.add_file(args.config_file)
    wandb.log_artifact(config_artifact)

    model_dir = c['model']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    data_files = c['dataset'].pop('data_files')
    training_data = Generator(data_files['training'], **c['dataset'])
    validation_data = Generator(data_files['validation'], **c['dataset'])
    # test_data = Generator(data_files['test'], **c['dataset'])

    input_shape = training_data.input_shape
    output_shape = training_data.output_shape
    if c['model']['predefined_model']:
        model = models.load_model(c['model']['predefined_model'])
    else:
        model = get_model(input_shape, output_shape, c['model'])

    print(model.summary())
    model.compile(
        loss=c['training']['loss'],
        metrics=c['training']['metrics'],
        optimizer=c['training']['optimizer']
    )
    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=c['training']['epochs'],
        callbacks=[wandb.keras.WandbCallback(save_model=False)]
    )
    save_results(model, history, model_dir)
    # model.evaluate()
    plot_results(model, history, model_dir, c['plotting'])


if __name__ == "__main__":
    main()

