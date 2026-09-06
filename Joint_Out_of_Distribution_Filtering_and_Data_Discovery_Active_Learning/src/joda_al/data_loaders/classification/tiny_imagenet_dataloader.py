import itertools
import os
from collections import defaultdict

from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder, VisionDataset
from torch import tensor, long
import numpy as np
from typing import Tuple, Any

from tqdm import tqdm
from torchvision import transforms as T
from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter
from joda_al.data_loaders.dataset_lib import PropertySubset


def load_tinyimagenet_pool(data_path, order, type=""):
    dataset_subpath = "tiny-imagenet-200"
    train_path=os.path.join(data_path,dataset_subpath,"train")
    val_path = os.path.join(data_path, dataset_subpath)
    transforms= T.ToTensor()
    training_set = TinyImageNet_train(train_path,transform=transforms)
    train_idx = list(itertools.chain.from_iterable([class_idx[:40] for key, class_idx in training_set.index_map.items()]))
    pool_idx = list(itertools.chain.from_iterable([class_idx[40:-50] for key, class_idx in training_set.index_map.items()]))
    val_idx = list(itertools.chain.from_iterable([class_idx[-50:] for key, class_idx in training_set.index_map.items()]))
    #Create Subpools and reindex
    training_pool = PropertySubset(training_set, train_idx+pool_idx)
    validation_set =  PropertySubset(training_set, val_idx)
    train_re_idx = list(range(len(train_idx)))
    pool_re_idx = list(range(len(train_idx),len(train_idx)+len(pool_idx)))
    test_set = TinyImageNet_val(val_path,transform=transforms,class_to_ids=training_set.class_to_idx)
    return training_pool, train_re_idx, [pool_re_idx], validation_set, test_set, TinyImageNetScenarioConverter()

class TinyImageNetScenarioConverter(DatasetScenarioConverter):

    @staticmethod
    def convert_task_to_pool_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set):
        return training_pool, label_idx, unlabeled_idcs[0], validation_set, test_set

    @staticmethod
    def convert_task_to_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set, num_streams=10):
        unlabeled_idcs = [unlabeled_idcs[0][i * (len(unlabeled_idcs[0]) // num_streams): (i + 1) * (len(unlabeled_idcs[0]) // num_streams)] for i in range(0, num_streams)]
        return training_pool, label_idx, unlabeled_idcs, validation_set, test_set

class TinyImageNet_train(ImageFolder):
    def __init__(self, file_path, transform=None):
        super().__init__(file_path, transform)
        self.data_map = defaultdict(list)
        self.index_map = defaultdict(list)
        for idx, (sample, target) in enumerate(zip(self.samples, self.targets)):
            self.index_map[target].append(idx)
            self.data_map[target].append(sample)
        self.data_samples = self.load_data()

    def load_data(self):
        data_samples=[]
        for data_path, _ in tqdm(self.samples,disable=True):
            img = Image.open(data_path).convert("RGB")
            data_samples.append(np.asarray(img).copy())
        return data_samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.data_samples[index], self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def get_class_counts(self, index):
        values, counts = np.unique(np.array(self.targets)[index], return_counts=True)
        assert np.all(values == sorted(values))  # should be sorted
        return counts


class TinyImageNet_val(VisionDataset):
    def __init__(self, file_path: str, transform=None, class_to_ids=None):
        super().__init__(file_path, transform)
        self.transform = transform
        if class_to_ids is None:
            class_to_ids=self.extract_labels_map(file_path)
        self.labelmap=class_to_ids
        self.targets, self.samples= self.load_data(os.path.join(file_path,"val"))


    def load_data(self,path):
        targets, images = self.extract_labels(os.path.join(path,"val_annotations.txt"))
        samples=[]
        targets= [self.labelmap[target] for target in targets]
        for img_file in images:
            with open(os.path.join(path,"images",img_file), "rb") as f:
                img = Image.open(f)
                samples.append(img.convert("RGB"))
        return targets, samples

    @staticmethod
    def extract_labels(file_path):
        targets = []
        images = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text_data=line.split("\t")  # strip() removes leading/trailing whitespace
                targets.append(text_data[1])
                images.append(text_data[0])
        return targets, images

    @staticmethod
    def extract_labels_map(file_path):
        targets = []
        images = []
        with open(os.path.join(file_path,"wnids.txt"), 'r', encoding='utf-8') as file:
            for i,line in enumerate(file):
                text_data=line.strip() # strip() removes leading/trailing whitespace
                targets.append(text_data[1])
                images.append(text_data[0])
        return targets, images

    def __getitem__(self, index):
        img = self.samples[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)

    def get_class_counts(self, index):
        values, counts = np.unique(np.array(self.targets)[index], return_counts=True)
        assert np.all(values == sorted(values))  # should be sorted
        return counts
