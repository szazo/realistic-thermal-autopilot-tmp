from typing import Any
import os
import tempfile
from pathlib import Path
import torch
from ..experiment_logger import ExperimentLoggerInterface


def load_torch_weights(
        path: Path,
        map_location: torch.device | None = None) -> dict[str, Any]:
    loaded = torch.load(path, map_location=map_location)
    return loaded


# save/load weights using experiment logger
class ExperimentLoggerWeightStore:

    _experiment_logger: ExperimentLoggerInterface

    def __init__(self, experiment_logger: ExperimentLoggerInterface):
        self._experiment_logger = experiment_logger

    def load_best_weights(
            self,
            map_device: torch.device | None = None,
            key_include_filename: bool = False) -> dict[str, Any]:

        return self.load_weights(checkpoint_name='best_weights',
                                 map_device=map_device,
                                 key_include_filename=key_include_filename)

    def load_weights(self,
                     checkpoint_name: str,
                     map_device: torch.device | None = None,
                     key_include_filename: bool = False) -> dict[str, Any]:

        with tempfile.TemporaryDirectory() as tmp:
            target_path = Path(os.path.join(tmp, 'weights_to_load.pth'))

            print('downloading weights:', checkpoint_name)
            self.download_weights(checkpoint_name=checkpoint_name,
                                  target_path=target_path,
                                  key_include_filename=key_include_filename)

            return load_torch_weights(target_path, map_location=map_device)

    def download_weights(self, checkpoint_name: str, target_path: Path,
                         key_include_filename: bool):

        key = f'checkpoint/{checkpoint_name}'
        if key_include_filename:
            key += f'/{checkpoint_name}.pth'

        self._experiment_logger.query_file(key,
                                           destination_path=str(target_path))

    def save_best_weights(self,
                          state_dict: dict[str, Any],
                          log_key: str,
                          target_filepath: str,
                          log=True):
        torch.save(state_dict, target_filepath)
        if log:
            self._experiment_logger.log_file(log_key, target_filepath)
