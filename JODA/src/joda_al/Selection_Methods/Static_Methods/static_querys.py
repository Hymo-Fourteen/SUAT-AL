from typing import Dict, Union, Optional, Tuple, List
import numpy as np
import random

from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.Selection_Methods.query_method_def import QueryMethod
from joda_al.models.model_lib import ModelHandler
from joda_al.task_supports.pytorch.TaskHandler import TaskHandler


class RandomQuery(QueryMethod):
    keywords=["Random"]

    @staticmethod
    def query(models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:
        unlabeled_idx_set = unlabeled_idx_set if unlabeled_idx_set is not None else dataset_handler.unlabeled_idcs
        indices = random.sample(unlabeled_idx_set, query_size)
        pool_values=[]
        return pool_values, indices


class FixedStepQuery(QueryMethod):
    keywords=["Fixed"]

    @staticmethod
    def query(models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:
        unlabeled_idx_set = unlabeled_idx_set if unlabeled_idx_set is not None else dataset_handler.unlabeled_idcs
        sampled = np.round(np.linspace(0, len(unlabeled_idx_set) - 1, query_size)).astype(int)
        indices = np.array(unlabeled_idx_set)[sampled]
        pool_values = []
        return pool_values, indices


class SumQuery(QueryMethod):
    keywords=["Sum"]

    @staticmethod
    def query(models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:
        unlabeled_idx_set = unlabeled_idx_set if unlabeled_idx_set is not None else dataset_handler.unlabeled_idcs
        if len(unlabeled_idx_set) > query_size:
            raise IndexError("Sum method expects query size to identical to unlabeled_idx_set length")
        pool_values = []
        return pool_values, unlabeled_idx_set


class AllQuery(QueryMethod):
    keywords = ["All"]

    @staticmethod
    def query(models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs) -> Tuple[List, List]:
        unlabeled_idx_set = unlabeled_idx_set if unlabeled_idx_set is not None else dataset_handler.unlabeled_idcs
        pool_values = []
        return pool_values, unlabeled_idx_set