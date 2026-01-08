import logging
import json
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import tyro

from trainer.common import ExperimentLoggerParameterStore, ExperimentLoggerWeightStore
from trainer.experiment_logger.config import MLFlowExperimentLoggerConfig, NeptuneExperimentLoggerConfig
from trainer.experiment_logger import ExperimentLoggerInterface, create_experiment_logger


class ModelDownloader:

    _log: logging.Logger

    _exp_logger: ExperimentLoggerInterface

    def __init__(self, exp_logger: ExperimentLoggerInterface):

        self._log = logging.getLogger(__class__.__name__)

        self._exp_logger = exp_logger

    def download_parameters(self, target_dir: Path):
        self._download_parameters(target_dir=target_dir, model_key='model')

    def download_weights(self, target_dir: Path, checkpoint_name: str,
                         weights_key_include_filename: bool):

        weights_path = target_dir / f'{checkpoint_name}.pth'
        self._download_weights(
            target_path=weights_path,
            checkpoint_name=checkpoint_name,
            weights_key_include_filename=weights_key_include_filename)

    def _download_parameters(self, target_dir: Path, model_key: str):

        self._log.debug(
            'downloading model parameters from the experiment logger...')

        # download the parameters
        parameter_store = ExperimentLoggerParameterStore(
            experiment_logger=self._exp_logger)

        parameters = parameter_store.load_parameters()

        model_parameters = parameters[model_key]
        self._log.debug('parameters loaded; model_parameters=%s',
                        model_parameters)

        # save as json
        with open(target_dir / 'model.json', 'w') as f:
            json.dump(model_parameters, f, indent=4)
        with open(target_dir / 'parameters.json', 'w') as f:
            json.dump(parameters, f, indent=4)

    def _download_weights(self, target_path: Path, checkpoint_name: str,
                          weights_key_include_filename: bool):

        # load weights
        self._log.debug('loading weights...; checkpoint_name=%s',
                        checkpoint_name)
        weights_store = ExperimentLoggerWeightStore(
            experiment_logger=self._exp_logger)

        weights_store.download_weights(
            checkpoint_name=checkpoint_name,
            target_path=target_path,
            key_include_filename=weights_key_include_filename)


@dataclass
class Epochs:
    run_id: str
    epochs: list[int]
    target_dir: str


@dataclass
class Checkpoint:
    run_id: str
    checkpoint_name: str
    target_dir: str


def create_neptune_logger(run_id: str):

    config_path = Path('config/birds/logger/neptune_logger.yaml')
    loaded_config = OmegaConf.load(config_path)

    assert isinstance(loaded_config, DictConfig)

    project = loaded_config['project']
    logger_config = NeptuneExperimentLoggerConfig(with_existing_id=run_id,
                                                  readonly=True,
                                                  project=project)

    experiment_logger = create_experiment_logger(logger_config)
    return experiment_logger


def create_mlflow_logger(run_id: str):

    config_path = Path('config/birds/logger/mlflow_logger.yaml')
    loaded_config = OmegaConf.load(config_path)

    assert isinstance(loaded_config, DictConfig)

    project = loaded_config['project']
    logger_config = MLFlowExperimentLoggerConfig(with_existing_id=run_id,
                                                 readonly=True,
                                                 project=project)

    experiment_logger = create_experiment_logger(logger_config)
    return experiment_logger


def main(config: Checkpoint | Epochs):

    if config.run_id.startswith('GLID'):
        # logged using neptune.ai
        experiment_logger = create_neptune_logger(run_id=config.run_id)
        weights_key_include_filename = False
    else:
        # mlflow
        experiment_logger = create_mlflow_logger(run_id=config.run_id)
        weights_key_include_filename = True

    target_dir_path = Path(config.target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    downloader = ModelDownloader(experiment_logger)

    # download the parameters
    downloader.download_parameters(target_dir_path)

    # download the weights
    if isinstance(config, Checkpoint):
        # single checkpoint
        downloader.download_weights(
            target_dir=target_dir_path,
            checkpoint_name=config.checkpoint_name,
            weights_key_include_filename=weights_key_include_filename)
    else:
        for epoch in config.epochs:
            checkpoint_name = f'weights{epoch}'
            downloader.download_weights(
                target_dir=target_dir_path,
                checkpoint_name=checkpoint_name,
                weights_key_include_filename=weights_key_include_filename)


if __name__ == '__main__':
    config = tyro.cli(Checkpoint | Epochs)
    main(config)
