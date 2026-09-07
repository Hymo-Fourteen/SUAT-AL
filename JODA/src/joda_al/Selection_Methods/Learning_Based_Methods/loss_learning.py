from typing import Dict, Union, Optional, Tuple, List
from warnings import warn
import numpy as np
import torch
from torch.utils.data import DataLoader

from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.data_loaders.sampler import SubsetSequentialSampler
from joda_al.Selection_Methods.query_method_def import QueryMethod
from joda_al.models.model_lib import ModelHandler
from joda_al.task_supports.pytorch.TaskHandler import TaskHandler


class LossLearningQuery(QueryMethod):
    keywords=["lloss"]

    @classmethod
    def query(cls, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs) -> Tuple[List, List]:
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = dataset_handler.get_unlabeled_pool_loader(unlabeled_idx_set)
        uncertainty = handler.get_predicted_loss(models, unlabeled_loader, device)
        # Measure uncertainty of each data points in the subset
        # uncertainty = get_predicted_loss(model, unlabeled_loader,device)
        dataset = dataset_handler.train_pool
        uncertainty, indices = cls.uncertainty_to_indices(uncertainty,dataset,unlabeled_idx_set)
        #indices = np.argsort(uncertainty)[::-1]
        set_indices = [unlabeled_idx_set[i] for i in indices]
        return uncertainty, set_indices

    @staticmethod
    def uncertainty_to_indices(uncertainty,dataset,unlabeled_idx_set):
        """
        Returns importance of samples in ascending order
        :param uncertainty:
        :return:
        """
        indices = np.argsort(uncertainty)[::-1]
        return uncertainty, indices


def get_predicted_loss(models, unlabeled_loader,device):
    models["task"].eval()
    models['module'].eval()
    with torch.cuda.device(device):
        uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            with torch.cuda.device(device):
                inputs = inputs.cuda()
            _, _, features = models["task"](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu().numpy()
