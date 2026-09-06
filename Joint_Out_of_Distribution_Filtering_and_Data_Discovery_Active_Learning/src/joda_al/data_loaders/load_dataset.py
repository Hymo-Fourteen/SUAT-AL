import itertools
from typing import Dict, Tuple, List, Union

import torch
from torch.utils.data import DataLoader, Dataset

from joda_al.Config.config_def import Config
from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter
from joda_al.Pytorch_Extentions.dataset_extentions import TransformDataset
from joda_al.defintions import TaskDef

from joda_al.data_loaders.SemanticSegmentation.load_segmentation_dataset import load_segsem_dataset, \
    get_training_transformations_semseg
from joda_al.data_loaders.classification.load_classification_dataset import load_classification_dataset, \
    get_training_transformations_classification, AlteredDataset
from joda_al.utils.logging_utils.log_writers import gl_info
from joda_al.utils.configmap import ConfigMap


def get_cls_counts(dataset, index :List[int], get_all_classes = False,remaped=True):
    """
    Get class counts for a given dataset and index.

    Args:
        dataset (Dataset): The dataset to get class counts from.
        index (int): The index list to get class counts for.
        get_all_classes (bool, optional): Whether to get counts for all classes. Defaults to False.
        remapped (bool, optional): Whether to remap the class counts. Defaults to True.

    Returns:
        Dict[int, int]: A dictionary with class counts.
    """
    gl_info(dataset.__class__)
    if (isinstance(dataset, torch.utils.data.Subset)):
        if (isinstance(dataset.dataset, torch.utils.data.Subset)):
            counts = dataset.dataset.dataset.get_class_counts(index)
        elif (isinstance(dataset.dataset, torch.utils.data.ConcatDataset)):
            raise NotImplementedError("Fixed cls_count")
            counts = {k: len(dataset) / 10 for k in range(10)}
        else:
            counts = dataset.dataset.get_class_counts(index)
    elif (isinstance(dataset, TransformDataset)):
        if (isinstance(dataset.dataset, torch.utils.data.Subset)):
            counts = dataset.dataset.dataset.get_class_counts(index)
        elif (isinstance(dataset.dataset, torch.utils.data.ConcatDataset)):
            raise NotImplementedError("Fixed cls_count")
            counts = {k: len(dataset) / 10 for k in range(10)}
        elif (isinstance(dataset.dataset, AlteredDataset)):
            counts = dataset.dataset.get_class_counts(index, remaped)
            if not get_all_classes:
                counts = {k: v for k, v in counts.items() if k in list(range(0, dataset.num_classes))}
        else:
            counts = dataset.dataset.get_class_counts(index)
    elif (isinstance(dataset, AlteredDataset)):
        counts = dataset.get_class_counts(index,remaped)
        if not get_all_classes:
            counts = {k:v for k, v in counts.items() if k in list(range(0, dataset.num_classes))}
    elif (isinstance(dataset.dataset.dataset, torch.utils.data.ConcatDataset)):
        raise NotImplementedError("Fixed cls_count")
        counts = {k: len(dataset)/10 for k in range(10)}
    else:
        counts = dataset.get_class_counts(index)
    return counts


def load_task_dataset(dataset: Dict, task : TaskDef, config : Union[Config,Dict], order=1) -> Tuple[Dataset, List[int],List[int],Dataset,Dataset, Dict, DatasetScenarioConverter]:
    """
    Loads the dataset for a given task.

    Args:
        dataset (Dict): The dataset configuration.
        task (TaskDef): The task to be performed on the dataset.
        config (Dict): The configuration for the dataset.
        order (int, optional): The order of the dataset. Defaults to 1.

    Returns:
        Tuple[Dataset, List[int], List[int], Dataset, Dataset, Dict, DatasetScenarioConverter]:
            - Dataset: The dataset for the given task.
            - List[int]: List of initially labeled indices for the training pool.
            - List[int]: List of unlabeled indices for the training pool.
            - Dataset: The training set.
            - Dataset: The validation set.
            - Dict: Additional dataset information.
            - DatasetFactory: The dataset factory.
    """
    gl_info("Load Dataset")
    if task == TaskDef.classification:
        return load_classification_dataset(dataset, config, order)
    elif task == TaskDef.semanticSegmentation:
        return load_segsem_dataset(dataset, config, order)
    else:
        raise NotImplementedError()


def get_training_transformations(task: TaskDef, dataset_name: str):
    """
    Get training transformations for a given task and dataset.

    Args:
        task (TaskDef): The task to be performed.
        dataset_name (str): The name of the dataset.

    Returns:
        Transformations for the given task and dataset.
    """
    if task == TaskDef.classification:
        return get_training_transformations_classification(dataset_name)
    elif task == TaskDef.semanticSegmentation:
        return get_training_transformations_semseg(dataset_name)
    else:
        raise NotImplementedError(f"{task} no supported")


def get_eval_transformations(task: TaskDef, dataset_name: str):
    """
    Get evaluation transformations for a given dataset and task.

    Args:
        dataset_name (str): The name of the dataset.
        task (TaskDef): The task to be performed.

    Returns:
        Transformations for evaluation.
    """
    pass


def load_pool_dataset(config: Union[Dict,ConfigMap]) -> Tuple[Dataset, List[int], List[int], Dataset, Dataset, Dict]:
    """
        Load a pool dataset based on the experiment configuration.

        Args:
            config (Dict): The experiment configuration.

        Returns:
            Tuple[Dataset, List[int], List[int], Dataset, Dataset, Dict]:
                - Dataset: The training pool.
                - List[int]: List of labeled indices.
                - List[int]: List of unlabeled indices.
                - Dataset: The validation set.
                - Dataset: The test set.
                - Dict: Additional dataset information.
    """
    dataset = {}
    dataset["name"] = config.experiment_config["dataset"]
    dataset["path"] = config.experiment_config["dataset_path"]
    dataset["static_configuration"]= config.experiment_config["static_configuration"]
    order=config.experiment_config["order"]
    task=config.experiment_config["task"]
    training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, factory = load_task_dataset(dataset, task, config, order)
    training_pool, label_idx, unlabeled_idx, validation_set, test_set = factory.convert_task_to_pool_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set)
    return training_pool, label_idx, unlabeled_idx, validation_set, test_set, dataset_config


def load_stream_dataset(config: Union[Dict,ConfigMap]) -> Tuple[Dataset, List[int], List[List[int]], Dataset, Dataset, Dict]:
    """
    Load a stream dataset based on the experiment configuration.

    Args:
        config (Dict): The experiment and data configuration.

    Returns:
        Tuple[Dataset, List[int], List[List[int]], Dataset, Dataset, Dict]:
            - Dataset: The training pool.
            - List[int]: List of labeled indices.
            - List[List[int]]: List of unlabeled indices for each stream.
            - Dataset: The validation set.
            - Dataset: The test set.
            - Dict: Additional dataset information.
    """
    dataset = {}
    dataset["name"] = config.experiment_config["dataset"]
    dataset["path"] = config.experiment_config["dataset_path"]
    dataset["static_configuration"]= config.experiment_config["static_configuration"]
    order=config.experiment_config["order"]
    task=config.experiment_config["task"]
    training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, factory = load_task_dataset(dataset, task, config, order)
    training_pool, label_idx, unlabeled_idcs, validation_set, test_set = factory.convert_task_to_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set)

    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config


def load_multi_stream_dataset(config: Union[Dict,ConfigMap]) -> Tuple[Dataset, List[int], List[List[int]], Dataset, Dataset, Dict]:
    """
    Load a multi-stream dataset based on the experiment configuration.

    Args:
        config (Dict): The experiment configuration.

    Returns:
        Tuple[Dataset, List[int], List[List[int]], Dataset, Dataset, Dict]:
            - Dataset: The training pool.
            - List[int]: List of labeled indices.
            - List[List[int]]: List of unlabeled indices for each stream.
            - Dataset: The validation set.
            - Dataset: The test set.
            - Dict: Additional dataset information.
    """
    dataset = {}
    dataset["name"] = config.experiment_config["dataset"]
    dataset["path"] = config.experiment_config["dataset_path"]
    dataset["static_configuration"]= config.experiment_config["static_configuration"]
    order=config.experiment_config["order"]
    task=config.experiment_config["task"]
    overlap=config.experiment_config["overlap"]
    training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, factory = load_task_dataset(dataset, task, config, order)
    if overlap !=0:
        training_pool, label_idx, unlabeled_idcs, validation_set, test_set = factory.convert_task_to_overlap_multi_stream_dataset(
            training_pool, label_idx, unlabeled_idcs, validation_set, test_set)

    else:
        training_pool, label_idx, unlabeled_idcs, validation_set, test_set = factory.convert_task_to_multi_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set)
    dataset_config["stream_lengths"] = [len(list(itertools.chain(*stream))) for stream in unlabeled_idcs]
    gl_info(factory)
    gl_info(f"Number of Selections: {len(unlabeled_idcs)}")
    for stream in unlabeled_idcs:
        gl_info(f"Number of agents {len(stream)}, {[len(s) for s in stream]}")
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config



