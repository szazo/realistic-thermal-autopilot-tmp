import tempfile
import json
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass
from mlflow import MlflowClient
import numpy as np
from flatten_dict import flatten
import json_tricks
import mlflow
from mlflow.entities import Experiment, Run

from .api import ExperimentLoggerInterface, ExperimentLoggerParametersBase
from .dictionary_to_yaml import dictionary_to_yaml


@dataclass(kw_only=True)
class MLFlowExperimentLoggerParameters(ExperimentLoggerParametersBase):
    project: str
    log_system_metrics: bool = True
    readonly: bool = False
    with_existing_id: str | None = None
    description: str | None = None


class MLFlowExperimentLogger(ExperimentLoggerInterface):

    _log: logging.Logger
    _params: MLFlowExperimentLoggerParameters

    _run: Run | None

    def __init__(
            self,
            project: str,
            log_system_metrics: bool = True,
            readonly: bool = False,
            with_existing_id: str | None = None,
            name: str | None = None,  # optional name of the run
            description: str | None = None) -> None:

        self._run = None
        self._log = logging.getLogger(__class__.__name__)

        self._params = MLFlowExperimentLoggerParameters(
            with_existing_id=with_existing_id,
            readonly=readonly,
            project=project,
            log_system_metrics=log_system_metrics,
            name=name,
            description=description)
        self._log.debug('initializing')

    def log_video(self, key: str, path: str):
        self.log_file(key, path)

    def log_file(self, key: str, path: str):
        run = self._resolve_run()
        mlflow.log_artifact(path, artifact_path=key, run_id=run.info.run_id)

    def query_file(self, key: str, destination_path: str):

        run = self._resolve_run()
        run_id = run.info.run_id

        target_file_path = Path(destination_path)
        target_dir = target_file_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        client = MlflowClient()
        client.download_artifacts(run_id=run_id,
                                  path=key,
                                  dst_path=str(target_dir))
        # rename
        downloaded_file_path = target_dir / Path(key).name
        downloaded_file_path.replace(target_file_path)

    def log_dict(self,
                 key: str,
                 dictionary: dict,
                 log_as_str=False,
                 log_as_pickle=False):

        self._log.debug('log_dict; key=%s,log_as_str=%s,log_as_pickle=%s', key,
                        log_as_str, log_as_pickle)

        run = self._resolve_run()
        run_id = run.info.run_id

        if log_as_pickle:
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir, f'{key}.pickle')
                with open(path, 'wb') as file:
                    # pickle the original dictionary
                    pickle.dump(dictionary, file)
                    file.flush()

                    mlflow.log_artifact(str(path), run_id=run.info.run_id)

        # fix the types for logging to yaml and mlflow
        fixed_types = json.loads(json_tricks.dumps(dictionary,
                                                   primitives=True))

        if log_as_str:
            yaml_string = dictionary_to_yaml(fixed_types)
            mlflow.log_text(artifact_file=f'{key}.yaml',
                            text=yaml_string,
                            run_id=run_id)

        # use the key as root
        dict_with_key = {key: fixed_types}

        # flatten
        flattened = flatten(dict_with_key, reducer='dot')

        mlflow.log_params(flattened, run_id=run_id)

    def query_dict_pickle(self, key: str, destination_path: str):
        self.query_file(key=f'{key}.pickle', destination_path=destination_path)

    def log_metrics(self,
                    key: str,
                    value: float | np.number | str,
                    step: int | None = None):
        self._log.debug('log_metrics; key=%s,value=%s,step=%s', key, value,
                        step)

        float_value: float = float(value)

        run = self._resolve_run()

        if isinstance(value, str):
            # REVIEW: remove string support
            mlflow.log_text(value,
                            artifact_file=f'{key}/step_{step}.txt',
                            run_id=run.info.run_id)
        else:
            mlflow.log_metric(key=key,
                              value=float_value,
                              run_id=run.info.run_id,
                              step=step)

    def _resolve_run(self) -> Run:

        if self._run is not None:
            return self._run

        experiment: Experiment = mlflow.set_experiment(
            experiment_name=self._params.project)

        if self._params.with_existing_id:
            run = mlflow.get_run(self._params.with_existing_id)
        else:
            run = mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=self._params.name,
                log_system_metrics=self._params.log_system_metrics,
                description=self._params.description)

        self._run = run
        return self._run

    def stop(self, success: bool):
        if self._run is None:
            return

        if self._params.readonly:
            return

        if success:
            mlflow.log_param(key='info/state', value='Success')
        else:
            mlflow.log_param(key='info/state', value='Failed')
            mlflow.log_param(key='sys/failed', value=True)

        mlflow.end_run()
