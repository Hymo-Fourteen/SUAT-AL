from typing import Dict, Union, List
import numpy as np
from tqdm import tqdm

# Custom
from joda_al.Selection_Methods.SelectionMethodFactory import SelectionMethodFactor
from joda_al.models.model_lib import ModelHandler, CompoundModel
from joda_al.utils.logging_utils.log_writers import gl_info
from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.Selection_Methods.method_regestry import LOSS_MODULE_METHODS
from joda_al.defintions import DataUpdateScenario
from joda_al.task_supports.pytorch.TaskHandler import TaskHandler
from joda_al.utils.logging_utils.training_logger import global_write_scalar
from joda_al.utils.configmap import ConfigMap


def factory_query(model: CompoundModel, method: str, dataset_handler: DataSetHandler, labeled_idx_set  : Union[List[int],None], unlabeled_idx_set : Union[List[int],None], query_size : int, device, config: Union[ConfigMap,Dict], handler:Union[TaskHandler,ModelHandler]=None, cycle_num=0):
    """
    Queries the dataset using the specified method and returns the selected pool values and indices.

    :param model: CompoundModel containing the model to be used for querying.
    :param method: String specifying the selection method to be used.
    :param dataset_handler: DataSetHandler object to handle the dataset.
    :param labeled_idx_set: List of indices of labeled data or None.
    :param unlabeled_idx_set: List of indices of unlabeled data or None.
    :param query_size: Integer specifying the number of samples to query.
    :param device: Device to be used for computation.
    :param config: Configuration map or dictionary.
    :param handler: TaskHandler or ModelHandler object, optional.
    :param cycle_num: Integer specifying the current cycle number, default is 0.
    :return: Tuple containing pool values and list of selected indices.
    """
    method=handle_loss_learning_extension(method)
    pool_values, indices = SelectionMethodFactor().create_or_get_query_method(method,**config.method.args).query(model, handler, dataset_handler,
              config, query_size, device,
              labeled_idx_set,
              unlabeled_idx_set)

    # Do not Clip to query size for All/ Identy selection
    if method =="All":
        return pool_values, list(indices)

    # OOD data filtering and precision calculation for Openset Active Learning
    if config["data_scenario"] in [DataUpdateScenario.osal, DataUpdateScenario.osal_extending]:
        inDindices, nD, fD = filter_osr(indices[:query_size], dataset_handler)
        gl_info(f"inD Share: {len(inDindices)/query_size}, nearOOD Share: {nD/query_size}, farOOD Share: {fD/query_size}")
        global_write_scalar("Selection/inD", len(inDindices)/query_size,cycle_num)
        global_write_scalar("Selection/nearOOD", nD/query_size,cycle_num)
        global_write_scalar("Selection/farOOD", fD/query_size,cycle_num)
        # Precision
        precision=(len(inDindices)+nD)/query_size
        global_write_scalar("Selection/Precision", precision,cycle_num)
        unlabeled_idx_set = unlabeled_idx_set if unlabeled_idx_set is not None else dataset_handler.current_unlabeled_idcs
        inD_targets=np.array(dataset_handler.train_pool.targets)[unlabeled_idx_set]
        unlabeled_inD=np.where(np.logical_and(np.array(inD_targets) < dataset_handler.dataset_config["num_classes"], np.array(inD_targets) > 0))[0]
        recall = (len(inDindices) + nD + len(dataset_handler.labeled_ind_idcs)) / (
                    len(unlabeled_inD) + len(dataset_handler.labeled_ind_idcs))
        gl_info(
            f"Selection Precision: {precision}, Recall(Lit): {recall}")
        global_write_scalar("Selection/Recall_Lit", recall,cycle_num)

    return pool_values, list(indices[:query_size])


def filter_osr(selected_idx_set, dh):
    filtered_indices = []
    nearOOD_counter = 0
    farOOD_counter = 0
    gl_info("Filtering closed set samples..")
    for idx in tqdm(selected_idx_set):
        _, label = dh.train_pool[idx]
        if label < dh.dataset_config["num_classes"] and label >= 0:
            filtered_indices.append(idx)
        elif int(label) >= dh.dataset_config["num_classes"]:
            nearOOD_counter += 1
        #elif 6 <= int(label) <= 7:
        #    nearOOD_counter += 1
        else:
            farOOD_counter += 1
    return filtered_indices, nearOOD_counter, farOOD_counter


def handle_loss_learning_extension(method):
    if method in LOSS_MODULE_METHODS:
        if method[-3:] == "Org":
            method = method[:-3]
        elif method[-1] == "L" and method not in ['VAAL', 'TA-VAAL']:
            method = method[:-1]
        elif method[-1] == "C":
            method = method[:-1]
        elif method[-4:] == "CMse":
            method = method[:-4]
        elif method[-4:] == "LOrg":
            method = method[:-4]
    return method