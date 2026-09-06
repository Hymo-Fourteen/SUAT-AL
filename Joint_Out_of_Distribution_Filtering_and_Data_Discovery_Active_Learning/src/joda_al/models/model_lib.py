from abc import abstractmethod, ABCMeta
from typing import List, Union, Tuple, Dict

from torch import nn

CompoundModel = Union[Dict[str,nn.Module],nn.ModuleDict]

class ModelHandler():
    ...

class EnsembleModel(nn.ModuleList):

    def get_model(self, idx):
        return self[idx]


class AbstractMcDropoutMixin(metaclass=ABCMeta):
    _mc_dropout_active = False

    @property
    def mc_dropout_active(self) -> bool:
        """
        :return: Returns channel number of features for loss learning module
        """
        return self._mc_dropout_active

    @mc_dropout_active.setter
    def mc_dropout_active(self,state) ->None:
        self._mc_dropout_active=state

    # @mc_dropout_active.getter
    # def mc_dropout_active(self) ->bool:
    #     return self._mc_dropout_active

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    @abstractmethod
    def set_dropout_status(self, status) -> None:
        """
        :param status: bool status to update dropout status
        """
        pass


class McDropoutMixin(AbstractMcDropoutMixin):

    _mc_dropout_active=False

    def __init__(self):
        """
        Pytroch Module does not call init, so do not define anything here
        """

    def get_do_status(self):
        if self._mc_dropout_active:
            return self._mc_dropout_active
        else:
            return self.training

    def set_dropout_status(self,status) -> None:
        self._mc_dropout_active=status


class McDropoutModelMixin(AbstractMcDropoutMixin):
    def __init__(self):
        self.mc_dropout_active = False

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def set_dropout_status(self, status):
        self.mc_dropout_active = status
        self.classifier.mc_dropout_active = status


class LossLearningMixin():
    _num_channels: List

    @property
    def num_channels(self) -> Union[List[int], List[List[int]]]:
        """
        :return: Returns channel number of features for loss learning module
        """
        return self._num_channels

    @num_channels.setter
    def num_channels(self, value):
        self._num_channels = value

    @abstractmethod
    def get_loss_learning_features(self, x: Tuple[int]) -> Union[List[int],List[List[int]]]:
        """
        :return: Feature Dimension for the features for loss learning module
        """
        pass
