from typing import Dict, Tuple, List

from torch import nn

from joda_al.Config.config_def import Config
from joda_al.task_supports.task_handler_factory import get_task_handler


class EngineOperator():

    def __init__(self, config):
        pass
        # task = config.experiment_config["task"]
        # self.task_handler = get_task_handler(task)

    @classmethod
    def train_and_evaluate(cls,
        task_handler,
        device,
        data_handler,
        dataset_config: Dict,
        config: Config,
        experiment_path: str,
        cycle_num: int,
    ) -> Tuple[List[float], nn.Module, Dict[str,float], List[Dict[str,float]]]:
        pass

    @classmethod
    def create_model_only(cls,
        task_handler,
        device,
        data_handler,
        dataset_config: Dict,
        config: Config,
        experiment_path: str,
        cycle_num: int,
        **kwargs
    ):
        pass


    @classmethod
    def unsupervised_training_only(cls,
                                   task_handler,
                                   device,
                                   data_handler,
                                   dataset_config: Dict,
                                   config: Config,
                                   experiment_path: str,
                                   cycle_num: int,
                                   **kwargs
                                   ):
        pass


