from typing import Type

from joda_al.Config.config_def import Config
from joda_al.data_loaders.data_handler.data_handler import DataSetHandler, OpenDataSetHandler, ExtOpenDataSetHandler
from joda_al.defintions import DataUpdateScenario


class DataSetHandlerFactory(object):
    """
    Factory class for creating dataset handlers based on the configuration.
    """
    def __new__(cls, config: Config) -> Type[DataSetHandler]:
        """
        Creates a new dataset handler based on the configuration.

        :param config: The configuration object.
        :return: A dataset handler class.
        """
        scenario: DataUpdateScenario = config.experiment_config["data_scenario"]
        if scenario == DataUpdateScenario.standard:
            return DataSetHandler
        elif scenario == DataUpdateScenario.osal:
            return OpenDataSetHandler
        elif scenario in [DataUpdateScenario.osal_extending, DataUpdateScenario.osal_near_far]:
            return ExtOpenDataSetHandler
        else:
            raise NotImplementedError(
                f"DataSetHandler for scenario {scenario} is not implemented. "
                "Please implement it in the DataSetHandlerFactory."
            )