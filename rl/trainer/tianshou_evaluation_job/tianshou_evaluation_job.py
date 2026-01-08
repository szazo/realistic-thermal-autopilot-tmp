from dataclasses import dataclass, asdict
from omegaconf import MISSING
import time
import logging
from ..experiment_logger import ExperimentLoggerConfigBase, ExperimentLoggerInterface
from ..base.tianshou_job_base import TianshouJobBase, TianshouJobParametersBase
from ..common import ExperimentLoggerParameterStore
from ..common.load_policy import PolicyCheckpoint, load_policy


@dataclass(kw_only=True)
class TianshouEvaluationJobParameters(TianshouJobParametersBase):
    device: str = 'cpu'
    evaluation_policy: PolicyCheckpoint
    logger: ExperimentLoggerConfigBase = MISSING


class TianshouEvaluationJob(TianshouJobBase):

    _params: TianshouEvaluationJobParameters

    def __init__(self, params: TianshouEvaluationJobParameters,
                 output_dir: str):

        super().__init__(output_dir=output_dir)

        self._log = logging.getLogger(__class__.__name__)
        self._params = params

    def evaluate(self):

        # create the logger for the evaluation
        experiment_logger = self._create_experiment_logger(
            self._params.logger, name_override=self._params.experiment_name)
        try:
            self._log.debug('evaluating (experiment_name=%s)...',
                            self._params.experiment_name)
            t1 = time.perf_counter(), time.process_time()

            policy, weights_state_dict = load_policy(
                source=self._params.evaluation_policy,
                device=self._params.device)

            # log the parameters
            ExperimentLoggerParameterStore(experiment_logger).save_parameters(
                asdict(self._params))

            # evaluate
            self._evaluate(policy=policy,
                           evaluators=self._params.evaluators,
                           experiment_logger=experiment_logger,
                           weights_state_dict=weights_state_dict)

            # run statistics
            self._run_statistics(statistics=self._params.statistics,
                                 source_logger=experiment_logger,
                                 target_logger=experiment_logger)

            t2 = time.perf_counter(), time.process_time()

            self._log.debug(
                f'evaluation finished; perf_counter: {t2[0]-t1[0]:.2f}s, process_time: {t2[1]-t1[1]:.2f}s'
            )
            self._finished(success=True, logger=experiment_logger)

        except Exception as e:
            self._log.error('error occurred during evaluation; exception=%s',
                            e)
            self._finished(success=False, logger=experiment_logger)
            raise

    def _finished(self, success: bool, logger: ExperimentLoggerInterface):

        self._log.debug('closing experiment logger...')
        logger.stop(success=success)
        self._log.debug('experiment logger closed')

        self._log.debug('job finished; success=%s', success)
