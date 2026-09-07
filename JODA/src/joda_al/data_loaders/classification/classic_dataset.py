import itertools
import os
import random
from collections import defaultdict
import numpy as np
import torch
from torchvision.datasets import Places365, MNIST, CIFAR10, CIFAR100
from typing import Tuple, Any, List, Optional, Callable
from torchvision.transforms import functional as F

from joda_al.utils.logging_utils.log_writers import gl_info

class UnbalancedDataset():
    def __init__(self,ind_classes,near_classes,validation,idx,ratio=1):
        label_map=defaultdict(list)
        idx_map = defaultdict(list)
        for i,(data,target) in enumerate(zip(self.data,self.target)):
            label_map[target].append(data)
            idx_map[target].append(data)
        for i in near_classes:
            idx_map[i]=idx_map[i][:int(len(idx_map[i])*ratio)]
        remaining_idx=itertools.chain(*idx_map.values())
        self.data=self.data[remaining_idx]
class UnbalancedCIFAR10(CIFAR10,UnbalancedDataset):
    def __init__(self):
        pass

class UnbalancedCIFAR100(CIFAR100):
    def __init__(self, root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        ind_classes : List =None,
        near_classes : List=None,
         total_classes:int=100,
        validation_idx : List=None,
        ratio=1):
        super().__init__(root, train, transform, target_transform, download)
        if near_classes is None or near_classes == []:
            near_classes = list(set(range(total_classes)) - set(ind_classes))
        label_map = defaultdict(list)
        idx_map = defaultdict(list)
        # for compatibility:
        filter_idx=validation_idx+[i for i in range(44000,45000)]
        for i, (data, target) in enumerate(zip(self.data, self.targets)):
            if  i in filter_idx:
                continue
            label_map[target].append(data)
            idx_map[target].append(i)
        for i in near_classes:
            idx_map[i] = idx_map[i][:int(len(idx_map[i]) * ratio)]
        # remaining_idx = random.shuffle(list(itertools.chain(*idx_map.values())))
        remaining_idces = list(itertools.chain(*idx_map.values()))
        remaining_idx = [i for i in range(len(self.targets)) if i in remaining_idces]
        remaining_set_len=len(remaining_idx)
        remaining_idx.extend(validation_idx)
        self.data = self.data[remaining_idx]
        self. targets = list(np.array(self.targets)[remaining_idx])

        self.label_idx= list(range(2000))
        unlabeled_set_len = remaining_set_len-2000
        equal_size=unlabeled_set_len//42
        unlabeled_idcs =  [list(range(i * equal_size+2000, (i + 1) * equal_size+2000)) for i in range(0, 42)]
        missing_size=unlabeled_set_len-equal_size*42
        unlabeled_idcs[-1]+=[ unlabeled_idcs[-1][-1]+i for i in range(missing_size)]
        self.unlabeled_idx = list(range(2000,remaining_set_len))
        self.unlabeled_idcs = unlabeled_idcs
        self.val_indices = list(range(remaining_set_len, remaining_set_len+len(validation_idx)))

class Cacheable_Places365(Places365):

    def __init__(self, root: str, *args, cache=True, **kwargs):
        super().__init__(root,*args,**kwargs)
        self.cache = cache
        if self.cache:
            self.data = self.load_cache_data(self.imgs)

    def load_cache_data(self,imgs):
        data=[]
        for img in imgs:
            data.append(self.transform(self.loader(img[0])))
        return data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not self.cache:
            return super().__getitem__(index)
        else:
            image, target = self.data[index], self.targets[index]

            return image, target


class Places365_ImageListDatset(Places365):

    def __init__(self, root: str, ind_dataset,*args, cache=True, **kwargs):
        super().__init__(root,*args,**kwargs)
        self.cache = cache
        self.filter_list = self.load_filter_file(ind_dataset)
        if self.cache:
            self.data, self.targets = self.load_cache_data(self.imgs,self.targets,self.filter_list)


    def load_cache_data(self,imgs,targets,filter):
        data=[]
        labels=[]
        for img, target in zip(imgs, targets):
            img_base=img[0].split('/')[-1]
            if img_base not in filter:
                continue
            data.append(self.transform(self.loader(img[0])))
            labels.append(target)
        return data, labels

    def load_filter_file(self,filter_set):
        if "cifar100" in filter_set:
            filter_file = "test_places365_cifar100.txt"
        elif "cifar10" in filter_set:
            filter_file = "test_places365_cifar10.txt"
        else:
            img_places = []
            return img_places
        gl_info(f"Using file {filter_file}")
        in_file=os.path.join(self.root,filter_file)
        with open(in_file, 'r') as f:
            lines = f.readlines()
        image_list = [line.strip() for line in lines]
        img_places = [img.split("/")[-1].split(" ")[0] for img in image_list]
        gl_info(f"Using {len(img_places)} images")
        return img_places
    
    
class Cacheable_Places365_Limited(Places365):

    def __init__(self, root: str, *args, cache=True, limit=200000, **kwargs):
        super().__init__(root,*args,**kwargs)
        self.cache = cache
        self.imgs, self.targets = self.imgs[:limit], self.targets[:limit]
        if self.cache:
            self.data = self.load_cache_data(self.imgs)

    def load_cache_data(self,imgs):
        data=[]
        for img in imgs:
            data.append(self.transform(self.loader(img[0])))
        return data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not self.cache:
            return super().__getitem__(index)
        else:
            image, target = self.data[index], self.targets[index]

            return image, target


class MNISTt2(MNIST):
    def __init__(self,*arg,size=40000,**kwargs):
        MNIST.__init__(self,*arg,**kwargs)
        data2, targets2 = self.double_dataset(self.data, self.targets,size)
        self.data = torch.concat([self.data,data2])
        self.targets = torch.concat([self.targets,targets2])


    @staticmethod
    def double_dataset(data,targets,size):
        new_data=F.rotate(data,90,expand=True)
        return new_data[:size],targets[:size]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")