import inspect

import numpy as np
from PIL import Image
import torch
import ast
import torchvision.transforms as tvs_trans
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision import transforms as T
import os
from collections import defaultdict

from joda_al.utils.configmap import ConfigMap


def pil_loader(img_str, str='RGB'):
    with Image.open(img_str) as img:
        img = img.convert(str)
    return img


INTERPOLATION = tvs_trans.InterpolationMode.BILINEAR


class OpenOODDataset(VisionDataset):
    def __init__(
            self,
            name,
            imglist_pth,
            data_dir,
            num_classes,
            preprocessor,
            data_aux_preprocessor,
            maxlen=None,
            dummy_read=False,
            dummy_size=None,
            **kwargs
        ):
            """
            Initialize the OODDataset object, used to import datasets from the OpenOOD library

            Args:
                name (str): The name of the dataset.
                imglist_pth (str): The path to the file containing the list of image names.
                data_dir (str): The directory where the image data is stored.
                num_classes (int): The number of classes in the dataset.
                preprocessor (object): The preprocessor object used to transform the images.
                data_aux_preprocessor (object): The preprocessor object used to transform auxiliary images.
                maxlen (int, optional): The maximum length of the samples. Defaults to None.
                dummy_read (bool, optional): Whether to perform dummy read. Defaults to False.
                dummy_size (int, optional): The size of the dummy read. Required if dummy_read is True.

            Raises:
                RuntimeError: If image_name starts with "/" when data_dir is not empty.
                ValueError: If dummy_read is True but dummy_size is not provided.
            """
            super(OpenOODDataset, self).__init__(**kwargs)

            self.name = name
            with open(imglist_pth) as imgfile:
                self.imglist = imgfile.readlines()
            self.data_dir = data_dir
            self.num_classes = num_classes
            self.preprocessor = preprocessor
            self.transform_image = preprocessor
            self.transform_aux_image = data_aux_preprocessor
            self.maxlen = maxlen
            self.dummy_read = dummy_read
            self.dummy_size = dummy_size

            self.samples = []
            for img_idx in range(len(self.imglist)):
                line = self.imglist[img_idx].strip('\n')
                tokens = line.split(' ', 1)
                if self.data_dir != '' and tokens[0].startswith('/'):
                    raise RuntimeError('image_name starts with "/"')
                self.samples.append(
                    (tokens[0], tokens[1], os.path.join(self.data_dir, tokens[0]), tokens)
                )

            if dummy_read and dummy_size is None:
                raise ValueError(
                    'if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def __getitem__(self, index):
        image_name, extra_str, path, tokens = self.samples[index]
        sample = {'image_name': image_name}
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        try:
            # some preprocessor methods require setup
            self.preprocessor.setup(**kwargs)
        except:
            pass

        try:
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                if not self.dummy_read:
                    image = pil_loader(path)
                sample['data'] = self.transform_image(image)

            sample['label'] = int(extra_str)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)

            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            raise e
        return sample["data"], sample["label"]


class OODDataset(VisionDataset):

    def __init__(self, dataset):
        for method in inspect.getmembers(dataset, predicate=inspect.ismethod):
            if method[0] not in self.__class__.__dict__.keys():
                self.__setattr__(method[0], method[1])
        for method in inspect.getmembers(dataset, predicate=inspect.isfunction):
            if method[0] not in self.__class__.__dict__.keys():
                self.__setattr__(method[0], method[1])
        for name, param in dataset.__dict__.items():
            if name not in self.__class__.__dict__.keys():
                self.__setattr__(name, param)
        self.dataset=dataset
        self.targets = [-1]*len(dataset)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and the mapped class index.
        """
        sample, _ = self.dataset.__getitem__(idx)
        return sample, self.targets[idx]

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.dataset.__len__()




class NoiseDataset(Dataset):
    def __init__(self, num_samples=10000, image_size=(3, 32, 32)):
        """
        Initializes a NoiseDataset object.

        Args:
            num_samples (int): The number of samples in the dataset. Default is 10000.
            image_size (tuple): The size of each image in the dataset. Default is (3, 32, 32).
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.targets = [-1]*num_samples

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and the label (-1 for noise).
        """
        return torch.randn(self.image_size), -1


class AlteredDataset(VisionDataset):
    root=None
    """
    A dataset class that alters the original dataset by mapping classes to new indices.

    Args:
        dataset (T): The original dataset.
        ind_classes (list): List of in distribution classes.
        near_classes (list): List of near ood classes.
        far_classes (list, optional): List of far ood classes. Defaults to an empty list.

    Attributes:
        dataset: The original dataset.
        num_classes: The total number of classes.
        map: A dictionary that maps classes to their corresponding indices.

    Methods:
        __getitem__(self, idx): Retrieves an item from the dataset.
        __len__(self): Returns the length of the dataset.
        get_class_counts(self, indices): Returns the class counts for a given set of indices.
    """

    def __init__(
        self, dataset: T, ind_classes: list, near_classes: list, far_classes: list = []
    ) -> None:
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
        self.targets=dataset.targets
        self.num_classes = len(ind_classes)
        indmap = dict(zip(ind_classes, range(len(ind_classes))))
        nearoodmap = dict(
            zip(near_classes, range(len(ind_classes), len(ind_classes) + len(near_classes)))
        )
        faroodmap = dict(zip(far_classes, [-1] * len(far_classes)))
        self.map = {**indmap, **nearoodmap, **faroodmap}

    def set_map(self,ind_classes,near_classes,far_classes):
        indmap = dict(zip(ind_classes, range(len(ind_classes))))
        nearoodmap = dict(
            zip(near_classes, range(len(ind_classes), len(ind_classes) + len(near_classes)))
        )
        faroodmap = dict(zip(far_classes, [-1] * len(far_classes)))
        self.map = {**indmap, **nearoodmap, **faroodmap}


    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and the mapped class index.
        """
        try:
            item = self.dataset.__getitem__(idx)
            if item is None:
                print(idx)
        except:
            print(idx)
        return (item[0], self.map[item[1]])

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.dataset.__len__()

    def get_class_counts(self, indices, remaped=True):
        """
        Returns the class counts for a given set of indices.

        Args:
            indices (list): List of indices.

        Returns:
            defaultdict: A dictionary containing the class counts.
        """
        if remaped:
            counts = defaultdict(int)
            for idx in indices:
                item = self.dataset.__getitem__(idx)
                counts[self.map[item[1]]] += 1
            return counts
        else:
            counts = defaultdict(int)
            for idx in indices:
                tar = self.dataset.targets[idx]
                counts[tar] += 1
            return counts
    

def get_attritubts(cls):
    import types
    k = [d for d in dir(cls) if not isinstance(getattr(cls, d), )]
    k=filter(lambda x: not x.startswith('__') and isinstance(getattr(cls, x), types.FunctionType), k)
    k=filter(lambda x: not x.startswith('__') and isinstance(getattr(cls, x), types.FunctionType), k)
    k=filter(lambda x: not x.startswith('__') and isinstance(getattr(cls, x), types.FunctionType), k)

default_preprocessing_dict = {
    'cifar6': {
        'pre_size': 32,
        'img_size': 32,
        'normalization': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    },
    'cifar50': {
        'pre_size': 32,
        'img_size': 32,
        'normalization': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    },
    'cifar10': {
        'pre_size': 32,
        'img_size': 32,
        'normalization': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    },
    'cifar100': {
        'pre_size': 32,
        'img_size': 32,
        'normalization': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    },
    'imagenet': {
        'pre_size': 256,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'imagenet200': {
        'pre_size': 256,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'TinyImageNet': {
        'pre_size': 64,
        'img_size': 64,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'TinyImageNetL': {
        'pre_size': 224,
        'img_size': 224,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'ImageNet800C': {
        'pre_size': 64,
        'img_size': 64,
        'normalization': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    },
    'aircraft': {
        'pre_size': 512,
        'img_size': 448,
        'normalization': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    },
    'cub': {
        'pre_size': 512,
        'img_size': 448,
        'normalization': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    }
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert(self.mode)
        else:
            return image.convert(self.mode)


class TestStandardPreProcessor():
    """For test and validation dataset standard image transformation."""
    def __init__(self, config: ConfigMap):
        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(config.pre_size, interpolation=INTERPOLATION),
            tvs_trans.CenterCrop(config.img_size),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(*config.normalization),
        ])

    def __call__(self, image):
        return self.transform(image)


def get_default_preprocessor(data_name: str):
    # TODO: include fine-grained datasets proposed in Vaze et al.?

    if data_name not in default_preprocessing_dict:
        raise NotImplementedError(f'The dataset {data_name} is not supported')

    config = ConfigMap(**default_preprocessing_dict[data_name])
    preprocessor = TestStandardPreProcessor(config)

    return preprocessor
