import os
from typing import List, Union, Callable

import torch
from PIL import Image
import json
from collections import Counter
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np

from joda_al.data_loaders.dataset_lib import get_mean, get_std, local_norm, AbstractActiveLearningDataset


class A2D2_Dataset(Dataset, AbstractActiveLearningDataset):

    def __init__(self, label_transforms:Union[Callable,None], path:Union[List[str],str],label_file="classification.json", *args,**kwargs):
        """
        Load Kitti dataset from the given path, using path as local path and path_server for the server path.
        The server bool switches between the two paths.
        :param transforms: Dataset transformations - resize of images for classification
        :param path: path for the session
        """
        # load all image files, sorting them to
        # ensure that they are aligned
        super().__init__(*args, **kwargs)
        self.transform=kwargs.get("transform",None)
        image_path, labels, dataset_timestamps, dataset_indicies = self.load_data(label_transforms,path,label_file)
        self.targets=labels
        self.dataset_indicies=dataset_indicies
        self.imgs = image_path
        self.timestamps=np.array(dataset_timestamps)
        assert len(image_path) == len(labels)

    def get_class_counts(self, index):
        values, counts = np.unique(np.array(self.targets)[index], return_counts=True)
        assert np.all(values == sorted(values))  # should be sorted
        return counts

    def load_data(self, label_transforms, paths: Union[List[str],str], label_file="classification.json"):
        """
        This function loads the data from the Kitti folder structure
        :param path: Path of the csv file of the images
        :return: Images and Class labels
        """
        if type(paths) == str:
            paths=[paths]
        dataset_img_paths = []
        dataset_labels = []
        dataset_indicies = []
        dataset_timestamps=[]
        for path in paths:
            print(path)
            session_img_paths=sorted([os.path.join(path,img_name) for img_name in list(filter(lambda p: (".jpg" in p or ".png" in p), os.listdir(path)))])

            if "json" in label_file:
                with open(path+"/"+label_file,"r") as f:
                    label = json.load(f)
            elif "txt" in label_file:
                import re
                labels=np.zeros((len(session_img_paths,)))
                actions = []
                label_path=path.replace("gtea_png/png","gtea_labels_61/old_labels")
                with open(label_path+".txt", 'r') as file:
                    lines = file.readlines()

                for line in lines:
                    # Use regular expressions to extract action labels and time intervals
                    # Assuming that each line follows the pattern: "<action1><action2> ... (start_time-end_time) [label]"
                    match = re.match(r'<(.+?)><(.+?)>\s*\((\d+)-(\d+)\)\s*\[(\d+)\]', line)

                    if match:
                        full_action_label = f"{match.group(1)}<{match.group(2)}>"
                        action_label = match.group(1)
                        object_label = match.group(2)
                        start_time = int(match.group(3))
                        end_time = int(match.group(4))
                        label = int(match.group(5))

                        # Append the parsed information as a tuple
                        actions.append((full_action_label, action_label, object_label, start_time, end_time, label))
                label=(labels,actions)
            else:
                label_path=path.replace("imgs","labels").replace("rgb-","")
                npzfile=np.load(label_path+label_file)
                labels, timestamps, start, end =npzfile["labels"],npzfile["timestamps"],npzfile["start_frame"],npzfile["end_frame"]
                image_timestamps=[int(s.split("_")[-1].split(".jpg")[0]) for s in session_img_paths]
                image_idx=[i for i,t in enumerate(image_timestamps) if t > timestamps[start] and t < timestamps[end]]
                session_img_paths=list(np.array(session_img_paths)[image_idx])
                filter_idx=[list(timestamps).index(ts) for ts in image_timestamps if ts > timestamps[start] and ts < timestamps[end]]
                label=labels[filter_idx][:,1]
               # np.any(label==0)
            if label_transforms is not None:
                label = label_transforms(label)
            dataset_img_paths += session_img_paths
            # Add indices before added new labels to reuse old length
            dataset_indicies.append([i for i in range(len(dataset_labels),len(dataset_img_paths))])

            dataset_labels+=[l for k,l in sorted(label.items())]
            if "cityscapes" in path:
                dataset_timestamps += [int(k.split("_")[-2]) for k, l in sorted(label.items())]
            elif "GTA-V-Streets" in path:
                dataset_timestamps += [int(k.split(".jpg")[0]) for k, l in sorted(label.items())]
            elif "Salad" in path:
                dataset_timestamps += image_timestamps
            elif "gtea" in path:
                dataset_timestamps += list(range(len(label)))
            else:
                dataset_timestamps+=[int(k.split("_")[-1].split(".png")[0]) for k,l in sorted(label.items())]
            if len(label.items()) != len(session_img_paths):
                print(f"path: {path} labeled incorrect labels {len(label.items())}, images {len(session_img_paths)}")

        return dataset_img_paths, dataset_labels, dataset_timestamps, dataset_indicies

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        img_tensor=transforms.ToTensor()(img)
        return img_tensor, self.targets[idx]

    def get_weights(self,idxs):
        """
        Calculate occurrence of the classes in the dataset for class weighted loss function
        :param idxs: list of indices for calculation
        :return: classwise weights
        """
        labels=[self.targets[idx] for idx in idxs]
        occurences=Counter(labels)
        sum_weigths = sum(occurences.values())
        weights = [sum_weigths/v for k, v in sorted(occurences.items())]
        return torch.tensor(weights)/np.linalg.norm(weights)

    def __len__(self):
        return len(self.imgs)


class A2D2_Dataset_Mem(A2D2_Dataset):
    def __init__(self, label_transforms, path: Union[List[str], str], *args, **kwargs):
        """
        Load Kitti dataset from the given path, using path as local path and path_server for the server path.
        The server bool switches between the two paths.
        :param transforms: Dataset transformations - resize of images for classification
        :param path: path for the session
        """
        # load all image files, sorting them to
        # ensure that they are aligned

        super().__init__(label_transforms, path,*args,**kwargs)
        self.imgs_array = self.load_images(self.imgs )

    @staticmethod
    def load_images(img_path_list):
        imgs=[]
        for img_path in img_path_list:
            img = Image.open(img_path).convert("RGB")
            imgs.append(np.asarray(img).copy())
        return np.array(imgs)

    def __getitem__(self, idx):
        img_tensor = transforms.ToTensor()(self.imgs_array[idx])
        return img_tensor, self.targets[idx]

    def get_image_visu(self,idx):
        return (transforms.ToTensor()(self.imgs_array[idx])).numpy()


class A2D2_Dataset_Mem_Norm(A2D2_Dataset_Mem):
    def __init__(self, label_transforms, path: Union[List[str], str], *args, **kwargs):
        super().__init__(label_transforms, path, *args, **kwargs)
        self.mean = []
        self.std = []

    def cal_mean_and_std(self,train_idx):
        self.mean= get_mean(self.imgs_array, train_idx)
        self.std= get_std(self.imgs_array, train_idx)
        return self.mean, self.std

    def set_mean_and_std(self,mean,std):
        self.mean=mean
        self.std=std

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        img_tensor = transform(self.imgs_array[idx])
        return img_tensor, self.targets[idx]


class A2D2_Dataset_Pre_Norm(A2D2_Dataset_Mem):
    def __init__(self, label_transforms, path: Union[List[str], str], *args, **kwargs):
        super().__init__(label_transforms, path, *args, **kwargs)

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        img_tensor = transform(self.imgs_array[idx])
        return img_tensor, self.targets[idx]


class A2D2_Dataset_Mem_Eq(A2D2_Dataset_Mem):
    def __init__(self, label_transforms, path: Union[List[str], str], *args, **kwargs):
        super().__init__(label_transforms, path, *args, **kwargs)

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            lambda x: transforms.F.equalize(x),
            transforms.ToTensor(),
        ])
        img_tensor = transform(self.imgs_array[idx])
        return img_tensor, self.targets[idx]


class A2D2_Dataset_Mem_LNorm(A2D2_Dataset_Mem):
    def __init__(self, label_transforms, path: Union[List[str], str], *args, **kwargs):
        super().__init__(label_transforms, path, *args, **kwargs)

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: local_norm(x),
        ])
        img_tensor = transform(self.imgs_array[idx])
        return img_tensor, self.targets[idx]
