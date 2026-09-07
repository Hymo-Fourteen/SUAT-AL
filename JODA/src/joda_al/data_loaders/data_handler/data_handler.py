from abc import abstractmethod
from typing import List, Optional, Dict, Union
import torch
from torch.utils.data import DataLoader, Subset

from joda_al.data_loaders.classification.load_classification_dataset import reinit_altered_propertysubset_set, \
    update_altered_dataset_map
from joda_al.data_loaders.load_dataset import get_training_transformations
from joda_al.data_loaders.sampler import SubsetSequentialSampler
from joda_al.Pytorch_Extentions.dataset_extentions import TransformSubset, TransformDataset
from joda_al.defintions import TaskDef
from joda_al.utils.logging_utils.log_writers import gl_warning, gl_info
from joda_al.utils.logging_utils.training_logger import seed_worker
from joda_al.data_loaders.load_dataset import get_cls_counts


class AbstractDataSetHandler():
    """
    Abstract base class for dataset handlers. Defines the required methods
    that all dataset handlers must implement.
    """
    @abstractmethod
    def get_test_loader(self)-> DataLoader:
        """Returns a DataLoader for the test dataset."""
        ...

    @abstractmethod
    def get_val_loader(self) -> DataLoader:
        """Returns a DataLoader for the validation dataset."""
        ...

    @abstractmethod
    def get_train_loader(self, labeled_idx: Optional[List[int]]=None) -> DataLoader:
        """Returns a DataLoader for the training dataset (labeled pool with augmentations)."""
        ...

    @abstractmethod
    def get_unlabeled_pool_loader(self, unlabeled_idx_set: Optional[List[int]]=None) -> DataLoader:
        """Returns a DataLoader for the unlabeled dataset."""
        pass

    @abstractmethod
    def get_full_pool_loader(self, unlabeled_idx_set: Optional[List[int]] = None, labeled_idx_set: Optional[List[int]]=None)-> DataLoader:
        """Returns a DataLoader for the full dataset (labeled + unlabeled)."""
        pass


class DataSetHandler(AbstractDataSetHandler):
    """
    Handles dataset operations such as loading, updating, and managing labeled/unlabeled pools.
    """

    unlabeled_idcs : List[int]
    labeled_idcs : List[int]

    def __init__(self, training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, config):
        """

        :param training_pool: torch.dataset
        :param label_idx: List[int]
        :param unlabeled_idcs: List[int]
        :param validation_set: torch.dataset
        :param test_set: torch.dataset
        :param dataset_config: Dict
        :param config: Dict
        """
        self.dataset_config = dataset_config
        self.train_pool = training_pool
        self.labeled_idcs = label_idx
        self.config = config
        task = self.config.experiment_config["task"]
        # Whole Unlabeled Datapool
        self.unlabeled_idcs = unlabeled_idcs
        # Current Unlabeled Datapool for Selection
        self.current_unlabeled_idcs = None
        if len(unlabeled_idcs) > 0  and type(unlabeled_idcs[0]) is not List:
            self.current_unlabeled_idcs=self.unlabeled_idcs

        # Uploaded but not labeled Datapool for Selection for two stage processes.
        self.selected_unlabeled_idcs = []

        self.validation_set=validation_set
        self.test_set = test_set

        if task is not None:
            self.init_task(task)
        # training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config

    def to_dict(self):
        """
        Converts the current state of the dataset handler to a dictionary.
        """
        status={}
        status["labeled_idcs"] = [int(i) for i in self.labeled_idcs]
        status["unlabeled_idcs"] = self.unlabeled_idcs
        status["selected_unlabeled_idcs"] = self.selected_unlabeled_idcs
        return status

    def load_from_dict(self, status):
        """
        Loads the state of the dataset handler from a dictionary.
        """
        self.labeled_idcs = status["labeled_idcs"]
        self.unlabeled_idcs = status["unlabeled_idcs"]
        self.selected_unlabeled_idcs = status["selected_unlabeled_idcs"]

    @classmethod
    def from_load_data(
        cls,
        training_pool: torch.utils.data.Dataset,
        label_idcx: List[int],
        unlabeled_idcs: List[int],
        validation_set: torch.utils.data.Dataset,
        test_set: torch.utils.data.Dataset,
        dataset_config: Dict[str, Union[str, int, List[int]]],
    ) -> "DataSetHandler":
        """
        Creates a new instance of the dataset handler from the given datasets and configurations.
        """
        gl_warning("Task and training config need to be set")
        return cls(training_pool, label_idcx, unlabeled_idcs, validation_set, test_set,dataset_config,None,None)

    def init_task(self, task: Union[TaskDef, str]):
        """
        Initializes the task-specific settings for the dataset handler.
        """
        self.task=task
        if task == TaskDef.objectDetection:
            fn_collate = detection_collate
        else:
            fn_collate = None
        self.fn_collate = fn_collate

    def get_test_loader(self) -> DataLoader:
        """
        Returns a DataLoader for the test dataset.
        """
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.config.training_config["batch_size"],
            num_workers=self.config.training_config["num_workers"],
            collate_fn=self.fn_collate,
        )
        return test_loader

    def get_val_loader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation dataset.
        """
        validation_loader = DataLoader(
            self.validation_set,
            pin_memory=True,
            batch_size=self.config.training_config["batch_size"],
            num_workers=self.config.training_config["num_workers"],
            collate_fn=self.fn_collate,
        )
        return validation_loader


    @staticmethod
    def get_transform_subset(dataset, indices, transform, transforms):
        return TransformSubset(
            dataset,
            indices,
            transform, transforms
            )

    @staticmethod
    def get_transform_dataset(dataset, transform, transforms):
        return TransformDataset(dataset, transform, transforms)

    def get_train_loader(self,labeled_idx: Optional[List[int]] = None,init_worker_seed: bool=False):
        """
        Returns a DataLoader for the training dataset.
        """
        if labeled_idx is None:
            labeled_idx=self.labeled_idcs
        training_set= self.get_transform_subset(self.train_pool,
            labeled_idx,
            **get_training_transformations(
                self.task, f'{self.dataset_config["name"]}-{self.dataset_config["dataset_scenario"]}'))
        if init_worker_seed:
            g = torch.Generator()
            g.manual_seed(self.config.training_config["seed"])
            training_loader = DataLoader(
                training_set,
                batch_size=self.config.training_config["batch_size"],
                shuffle=True,
                num_workers=self.config.training_config["num_workers"],
                drop_last=True,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            training_loader = DataLoader(
                training_set,
                batch_size=self.config.training_config["batch_size"],
                shuffle=True,
                num_workers=self.config.training_config["num_workers"],
                drop_last=True
            )
        return training_loader

    def get_unlabeled_pool_loader(self, unlabeled_idx_set=None, batch_size=None):
        if unlabeled_idx_set is None:
            unlabeled_idx_set = self.current_unlabeled_idcs
        if batch_size is None:
            batch_size = self.config.training_config["batch_size"]
        unlabeled_loader = DataLoader(self.train_pool, batch_size=batch_size,
                                      num_workers=self.config.training_config["num_workers"],
                                      sampler=SubsetSequentialSampler(unlabeled_idx_set),
                                      pin_memory=True, collate_fn=self.fn_collate)
        return unlabeled_loader

    def get_labeled_pool_loader(self, labeled_idx_set=None,batch_size=None):
        if labeled_idx_set is None:
            labeled_idx_set = self.labeled_idcs
        if batch_size is None:
            batch_size = self.config.training_config["batch_size"]
        labeled_loader = DataLoader(self.train_pool, batch_size=batch_size,
                                      num_workers=self.config.training_config["num_workers"],
                                      sampler=SubsetSequentialSampler(labeled_idx_set),
                                      pin_memory=True, collate_fn=self.fn_collate)
        return labeled_loader

    def get_full_pool_loader(
        self, unlabeled_idx_set: Optional[List[int]] = None, labeled_idx_set: Optional[List[int]] = None
    ) -> DataLoader:
        """
        Returns a DataLoader for the full dataset (labeled + unlabeled).
        """
        if unlabeled_idx_set is None:
            unlabeled_idx_set=self.current_unlabeled_idcs
        if labeled_idx_set is None:
            labeled_idx_set=self.labeled_idcs
        pool_loader = DataLoader(self.train_pool, batch_size=self.config.training_config["batch_size"],
                                      num_workers=self.config.training_config["num_workers"],
                                      sampler=SubsetSequentialSampler(unlabeled_idx_set + labeled_idx_set),
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True, collate_fn=self.fn_collate)
        return pool_loader


    def update_labeled_idx_set(self, idx_list: List[int]) -> None:
        """
        Updates the list of labeled indices.
        """
        self.labeled_idcs = idx_list

    def update_current_unlabeled_idx_set(self, idx_list: List[int]) -> None:
        """
        Updates the list of current unlabeled indices.
        """
        self.current_unlabeled_idcs=idx_list

    def update_collected_unlabeled_idx_set(self, idx_list: List[int]) -> None:
        """
        Updates the list of collected unlabeled indices.
        """
        self.selected_unlabeled_idcs=idx_list

    # def seed_workers(self):
    #     pass

    def _get_pool_loader(self, idx : List[int]) -> DataLoader:
        """
        Returns a DataLoader for a specific subset of the dataset.
        """
        pool_loader = DataLoader(self.train_pool, batch_size=self.config.training_config["batch_size"],
                                      num_workers=self.config.training_config["num_workers"],
                                      sampler=SubsetSequentialSampler(idx),
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True, collate_fn=self.fn_collate)
        return pool_loader

    def _get_trainable_pool_loader(self, idx: List[int], init_worker_seed: bool = False) -> DataLoader:
        """
        Returns a DataLoader for a specific subset of the dataset, with optional worker seeding.
        """
        training_set=self.get_sub_pool(idx)
        if init_worker_seed:
            g = torch.Generator()
            g.manual_seed(self.config.training_config["seed"])
            training_loader = DataLoader(
                training_set,
                batch_size=self.config.training_config["batch_size"],
                shuffle=True,
                num_workers=self.config.training_config["num_workers"],
                drop_last=True,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            training_loader = DataLoader(
                training_set,
                batch_size=self.config.training_config["batch_size"],
                shuffle=True,
                num_workers=self.config.training_config["num_workers"],
                drop_last=True
            )
        return training_loader

    def get_sub_pool(self,idx):
        return Subset(self.train_pool, idx)


class OpenDataSetHandler(DataSetHandler):

    def __init__(self, training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, config):
        """

        :param training_pool: torch.dataset
        :param label_idx: List[int]
        :param unlabeled_idcs: List[int]
        :param validation_set: torch.dataset
        :param test_set: torch.dataset
        :param dataset_config: Dict
        :param config: Dict
        """
        super().__init__(training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, config)
        self.labeled_ind_idcs = []
        self.labeled_far_ood_idcs = []
        self.update_labeled_idx_set(self.labeled_idcs)

    def update_data_pool(self, data_creator_function):
        # implement other bases
        cls_counts = get_cls_counts(self.train_pool, self.labeled_idcs, get_all_classes=True,remaped=False)
        new_classes = [k for k,v in cls_counts.items() if (v >= self.config.experiment_config["expanding_th"] and k not in self.config.experiment_config["ind_classes"] and k != -1)]

        self.config.experiment_config["ind_classes"] += new_classes
        self.config.experiment_config["near_classes"] = [c for c in self.config.experiment_config["near_classes"] if c not in new_classes]
        gl_info(f"New classes: {new_classes}, remaining Near classes: {self.config.experiment_config['near_classes']}, all counts: {cls_counts}")
        ind_classes = self.config.experiment_config["ind_classes"]
        # val_indices = [idx for idx in range(len(validation_set.dataset)) if validation_set.targets[idx] in ind_classes]
        # test_indices = [idx for idx in range(len(test_set.dataset)) if test_set.targets[idx] in ind_classes]

        # training_pool, _, _, validation_set, test_set, dataset_config = data_creator_function(self.config.experiment_config)
        # self.train_pool = training_pool

        # Update Training Pool Map
        update_altered_dataset_map(self.train_pool, self.config.experiment_config["ind_classes"], self.config.experiment_config["near_classes"],self.config.experiment_config["far_classes"])

        #Update Subset and Map of Validation and Test Set
        self.validation_set = reinit_altered_propertysubset_set(self.validation_set,self.config,self.dataset_config)
        self.test_set = reinit_altered_propertysubset_set(self.test_set,self.config,self.dataset_config)

        # Update Number of Classes for Model generation
        self.dataset_config["num_classes"] = len(self.config.experiment_config["ind_classes"])
        self.dataset_config["ind_classes"] = [int(i) for i in self.config.experiment_config["ind_classes"]]
        self.dataset_config["near_classes"] = [int(i) for i in self.config.experiment_config["near_classes"]]
        self.update_labeled_idx_set()

    def to_dict(self):
        """
        Converts the current state of the dataset handler to a dictionary.
        """
        status = super().to_dict()
        status["labeled_ind_idcs"] = [int(i) for i in self.labeled_ind_idcs]
        status["labeled_far_ood_idcs"] = [int(i) for i in self.labeled_far_ood_idcs]
        if "ind_classes" not in self.dataset_config:
            self.dataset_config["ind_classes"] = [int(i) for i in self.config.experiment_config["ind_classes"]]
        if "near_classes" not in self.dataset_config:
            self.dataset_config["near_classes"] = [int(i) for i in self.config.experiment_config["near_classes"]]
        status["dataset_config"] = self.dataset_config
        return status

    def load_from_dict(self, status):
        """
        Loads the state of the dataset handler from a dictionary.
        """
        super().load_from_dict(status)
        self.labeled_ind_idcs = status["labeled_ind_idcs"]
        self.labeled_far_ood_idcs = status["labeled_far_ood_idcs"]
        self.dataset_config = status["dataset_config"]
        self.config.experiment_config["ind_classes"] = self.dataset_config["ind_classes"]
        self.config.experiment_config["near_classes"] = self.dataset_config["near_classes"]

    def update_labeled_idx_set(self,idx_list):
        """
        Updates the list of labeled indices and estimates OOD and Discoverable samples.
        """
        self.labeled_idcs=idx_list
        self.labeled_far_ood_idcs = [idx for idx in idx_list if self.train_pool.targets[idx] not in self.config.experiment_config["ind_classes"]]
        self.labeled_ind_idcs = [idx for idx in idx_list if self.train_pool.targets[idx] in self.config.experiment_config["ind_classes"]]

    def get_labeled_ood_pool_loader(self, labeled_idx_set=None,batch_size=None):
        """
        Returns the currently collected and identified OOD samples.
        """
        if labeled_idx_set is None:
            labeled_idx_set = self.labeled_far_ood_idcs
        if batch_size is None:
            batch_size = self.config.training_config["batch_size"]
        labeled_loader = DataLoader(self.train_pool, batch_size=batch_size,
                                      num_workers=self.config.training_config["num_workers"],
                                      sampler=SubsetSequentialSampler(labeled_idx_set),
                                      pin_memory=True, collate_fn=self.fn_collate)
        return labeled_loader


class ExtOpenDataSetHandler(OpenDataSetHandler):

    def __init__(self, training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, config):
        super().__init__(training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, config)
        self.labeled_near_ood_idcs = []
        self.update_labeled_idx_set(self.labeled_idcs)

    def to_dict(self):
        """
        Converts the current state of the dataset handler to a dictionary.
        """
        status = super().to_dict()
        status["labeled_near_ood_idcs"] = [int(i) for i in self.labeled_near_ood_idcs]
        return status

    def load_from_dict(self, status):
        """
        Loads the state of the dataset handler from a dictionary.
        """
        super().load_from_dict(status)
        self.labeled_near_ood_idcs = status["labeled_near_ood_idcs"]

    def update_labeled_idx_set(self, idx_list: Optional[List] = None):
        """
        Updates the list of labeled indices and estimates OOD and Discoverable samples.
        :param idx_list optional list to update labeled indices, otherwise labels will be reassigned is novel classes have been discovered

        """
        if idx_list is None:
            idx_list = self.labeled_idcs
        else:
            self.labeled_idcs = idx_list
        self.labeled_near_ood_idcs = [idx for idx in idx_list if
                                 self.train_pool.targets[idx] in self.config.experiment_config["near_classes"]]
        self.labeled_far_ood_idcs = [idx for idx in idx_list if
                                 self.train_pool.targets[idx] in self.config.experiment_config["far_classes"]]
        self.labeled_ind_idcs = [idx for idx in idx_list if
                                 self.train_pool.targets[idx] in self.config.experiment_config["ind_classes"]]

    def get_labeled_ood_pool_loader(self, labeled_idx_set=None, batch_size=None):
        """
        Returns the currently collected and identified OOD samples.
        """
        if labeled_idx_set is None:
            labeled_idx_set = self.labeled_far_ood_idcs
        if batch_size is None:
            batch_size = self.config.training_config["batch_size"]
        labeled_loader = DataLoader(self.train_pool, batch_size=batch_size,
                                      num_workers=self.config.training_config["num_workers"],
                                      sampler=SubsetSequentialSampler(labeled_idx_set),
                                      pin_memory=True, collate_fn=self.fn_collate)
        return labeled_loader


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    #https://github.com/amdegroot/ssd.pytorch/blob/5b0b77faa955c1917b0c710d770739ba8fbff9b7/data/__init__.py#L9
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets


def zip_collate_fn(batch):
    return tuple(zip(*batch))
