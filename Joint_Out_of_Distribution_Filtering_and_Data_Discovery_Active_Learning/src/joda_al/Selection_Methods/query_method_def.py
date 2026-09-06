from abc import abstractmethod, ABC
from typing import Dict, Union, overload, Tuple, List, Optional

from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.models.model_lib import ModelHandler, CompoundModel
from joda_al.task_supports.pytorch.TaskHandler import TaskHandler


class QueryMethod(ABC):
    keywords = None
    def __init__(self, *args, **kwarg):
        self.cycles = 0
        pass

    @overload
    @staticmethod
    @abstractmethod
    def query(models: CompoundModel, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:
        ...
    """
    Query the  using the provided models and handlers on a dataset.

    Args:
        models (Dict): A dictionary of models to be used in the query.
        handler (Union[TaskHandler, ModelHandler]): An instance of a task handler or model handler.
        dataset_handler (DataSetHandler): An instance of a dataset handler.
        training_config (Dict): Configuration settings for the training process.
        query_size (int): The desired size of the query.
        device: The device on which the query will be executed (e.g., 'cpu', 'cuda').
        *arg: Variable-length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tuple[List, List]: A tuple containing two lists:
            - The first list contains Values justifying the selection.
            - The second list contains the unlabeled indices for selection sorted by importance in descending order.

    """

    @overload
    @classmethod
    @abstractmethod
    def query(cls, models: CompoundModel, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:
        ...
    """
    Query the  using the provided models and handlers on a dataset.

    Args:
        models (Dict): A dictionary of models to be used in the query.
        handler (Union[TaskHandler, ModelHandler]): An instance of a task handler or model handler.
        dataset_handler (DataSetHandler): An instance of a dataset handler.
        training_config (Dict): Configuration settings for the training process.
        query_size (int): The desired size of the query.
        device: The device on which the query will be executed (e.g., 'cpu', 'cuda').
        *arg: Variable-length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tuple[List, List]: A tuple containing two lists:
            - The first list contains Values justifying the selection.
            - The second list contains the unlabeled indices for selection sorted by importance in descending order.
    """


    @abstractmethod
    def query(self, models: CompoundModel, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:
        ...
    """
    Query the  using the provided models and handlers on a dataset.

    Args:
        models (Dict): A dictionary of models to be used in the query.
        handler (Union[TaskHandler, ModelHandler]): An instance of a task handler or model handler.
        dataset_handler (DataSetHandler): An instance of a dataset handler.
        training_config (Dict): Configuration settings for the training process.
        query_size (int): The desired size of the query.
        device: The device on which the query will be executed (e.g., 'cpu', 'cuda').
        *arg: Variable-length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tuple[List, List]: A tuple containing two lists:
            - The first list contains Values justifying the selection.
            - The second list contains the unlabeled indices for selection sorted by importance in descending order.
    """


class QueryFactory:
    queries = {}

    @classmethod
    def register_query(cls, definition: QueryMethod):
        ...
