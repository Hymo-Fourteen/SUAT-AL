import inspect
from abc import abstractmethod
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset
from torchvision import transforms as T
from torchvision.datasets import VisionDataset

# from torchvision import transforms

from joda_al.Pytorch_Extentions import transforms
from joda_al.defintions import PreProcessingDef


def update_means(dataset_name, train_idx, train_dataset, val_dataset, test_dataset):
    if dataset_name in MEANED_DATASETS:
        mean, std = train_dataset.cal_mean_and_std(train_idx)
        val_dataset.set_mean_and_std(mean, std)
        test_dataset.set_mean_and_std(mean, std)


MEANED_DATASETS = ["A2D2N", "A2D2sN", "A2D2mN", "A2D2lN", "A2D2NR", "A2D2NR2", "A2D2NR3", "nuscenesN", "nuscenesNR",
                   "cityscapesN2", "cityscapesNR", "A2D2LN", "A2D2mNR", "GTAVSN", "GTAVSmN"]


def get_mean(img_list, idx=None):
    """
    This function calculated the channelwise mean
    :param dataset: torch.dataset input to mean
    :param idx: indices for mean
    :return:
    """
    if idx is not None:
        subset = [img_list[i] for i in idx]
        image_array = np.array(subset)
    else:
        image_array = np.array(img_list)
    ch0_mean = np.mean(image_array[:, :, :, 0])/255
    ch1_mean = np.mean(image_array[:, :, :, 1])/255
    ch2_mean = np.mean(image_array[:, :, :, 2])/255
    return ch0_mean, ch1_mean, ch2_mean


def get_std(img_list, idx=None):
    """
    This function calculated the channelwise standard deviation
    :param dataset: torch.dataset input to mean
    :param idx: indices for mean
    :return:
    """
    if idx is not None:
        subset = [img_list[i] for i in idx]
        image_array = np.array(subset)
    else:
        image_array = np.array(img_list)
    ch0_std = np.std(image_array[:, :, :, 0]/255)
    ch1_std = np.std(image_array[:, :, :, 1]/255)
    ch2_std = np.std(image_array[:, :, :, 2]/255)
    return ch0_std, ch1_std, ch2_std


def local_norm(x):
    ch0_max = torch.max(x[0, :, :])
    ch1_max = torch.max(x[1, :, :])
    ch2_max = torch.max(x[2, :, :])
    ch0_min = torch.min(x[0, :, :])
    ch1_min = torch.min(x[1, :, :])
    ch2_min = torch.min(x[2, :, :])
    nl=transforms.Normalize([ch0_min,ch1_min,ch2_min], [ch0_max-ch0_min,ch1_max-ch1_min,ch2_max-ch2_min])
    return nl(x)


class DatasetPreprocessing():
    mean = None
    std = None
    def __init__(self,preprocessing):
        print(DatasetPreprocessing)
        if preprocessing == PreProcessingDef.norm:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RefNormalize(lambda: self.mean, lambda: self.std)
            ])

        elif preprocessing == PreProcessingDef.equalized:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                lambda x: transforms.F.equalize(x),
                transforms.ToTensor(),
            ])
        elif preprocessing == PreProcessingDef.local_maxed:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: local_norm(x),
            ])

        elif preprocessing == PreProcessingDef.pretrained_norm:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        elif preprocessing == PreProcessingDef.none:
            self.transform = transforms.ToTensor()


class AbstractActiveLearningDataset():
    """
    Abstract base class for active learning datasets.

    This class defines the interface for active learning datasets, requiring
    the implementation of the `get_class_counts` method in subclasses.
    """
    @abstractmethod
    def get_class_counts(self, index: List[int]):
        """
        Abstract method to get the class counts for the given indices.

        Args:
            index (List[int]): List of indices for which class counts are required.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Must be implemented in child class")


class ActiveLearningClassificationDataset(VisionDataset, AbstractActiveLearningDataset):
    """
    A wrapper class for active learning classification datasets.

    This class extends a given dataset to include functionality for active learning,
    such as retrieving class counts for specific indices.

    Attributes:
        root (str): Root directory of the dataset.
        dataset (VisionDataset): The wrapped dataset instance.
    """
    root = None
    dataset = None

    def __init__(self, dataset: T) -> None:
        """
        Initializes the ActiveLearningClassificationDataset.

        Args:
            dataset (VisionDataset): The dataset to wrap and extend.
        """
        for method in inspect.getmembers(dataset, predicate=inspect.ismethod):
            if method[0] not in self.__class__.__dict__.keys():
                self.__setattr__(method[0], method[1])
        for method in inspect.getmembers(dataset, predicate=inspect.isfunction):
            if method[0] not in self.__class__.__dict__.keys():
                self.__setattr__(method[0], method[1])
        for name, param in dataset.__dict__.items():
            if name not in self.__class__.__dict__.keys():
                self.__setattr__(name, param)
        self.dataset = dataset

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Any: The item at the specified index.
        """
        return self.dataset.__getitem__(idx)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return self.dataset.__len__()

    def __setattr__(self, key, value):
        """
        Sets an attribute for the dataset and propagates it to the wrapped dataset.

        Args:
            key (str): Attribute name.
            value (Any): Attribute value.
        """
        super().__setattr__(key, value)
        if self.dataset is not None:
            self.dataset.__setattr__(key, value)

    def get_class_counts(self, index):
        """
        Retrieves the class counts for the specified indices.

        Args:
            index (List[int]): List of indices for which class counts are required.

        Returns:
            np.ndarray: Array of class counts.
        """
        values, counts = np.unique(np.array(self.targets)[index], return_counts=True)
        assert np.all(values == sorted(values))  # should be sorted
        return counts


class PropertySubset(Subset):
    """
    A subset of a dataset with additional properties for targets and transforms.

    This class extends the `Subset` class to provide access to the targets
    and transform properties of the underlying dataset.
    """

    @property
    def targets(self):
        """
        Retrieves the targets for the subset.

        Returns:
            List[Any]: List of targets corresponding to the subset indices.
        """
        return [self.dataset.targets[i] for i in self.indices]

    @property
    def transform(self):
        """
        Retrieves the transform applied to the dataset.

        Returns:
            Callable: The transform function applied to the dataset.
        """
        return self.dataset.transform

    @transform.setter
    def transform(self, transform):
        """
        Sets the transform for the dataset.

        Args:
            transform (Callable): The transform function to apply to the dataset.
        """
        self.dataset.transform = transform


class PropertyConcatDataset(ConcatDataset):
    """
    A concatenated dataset with additional property for targets.

    This class extends the `ConcatDataset` class to provide access to the
    combined targets of all datasets.
    """

    def __init__(self, datasets: Iterable[VisionDataset]) -> None:
        """
        Initializes the PropertyConcatDataset.

        Args:
            datasets (Iterable[VisionDataset]): List of datasets to concatenate.
        """
        super().__init__(datasets)
        self.targets = []
        for dataset in datasets:
            self.targets.extend(dataset.targets)