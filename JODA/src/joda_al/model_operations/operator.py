from typing import List, Dict, Tuple

from torch import nn

from joda_al.Config.config_def import Config
from joda_al.defintions import Engine
from joda_al.model_operations.engine import EngineOperator
from joda_al.task_supports.task_handler_factory import get_task_handler


def get_operator(config: Config) -> EngineOperator:
    if config.experiment_config.get("engine", "pytorch") == Engine.pytorch:
        from joda_al.model_operations.pytorch.pytorch_engine import PyTorchOperator
        return PyTorchOperator(config)
    else:
        raise NotImplementedError(f"Only Pytorch is currently supported.")


class Operator():

    def __init__(self, config):
        task = config.experiment_config["task"]
        self.task_handler = get_task_handler(task, config.experiment_config["engine"])
        self.engine_operator = get_operator(config)

    def train_and_evaluate(self, *args, **kwargs) -> Tuple[List[float], nn.Module, Dict[str,float], List[Dict[str,float]]]:
        return self.engine_operator.train_and_evaluate(self.task_handler,*args,**kwargs)

    def create_model_only(self, *args, **kwargs) -> Tuple[List[float], nn.Module, Dict[str,float], List[Dict[str,float]]]:
        return self.engine_operator.create_model_only(self.task_handler,*args,**kwargs)

