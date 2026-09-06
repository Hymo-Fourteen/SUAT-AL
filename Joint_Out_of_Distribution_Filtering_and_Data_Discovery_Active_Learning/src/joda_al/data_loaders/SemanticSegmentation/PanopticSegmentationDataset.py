import os
from typing import Union, Callable, List, Optional
from warnings import warn

import numpy as np
import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset


from joda_al.data_loaders.dataset_lib import get_mean, get_std, DatasetPreprocessing
from joda_al.defintions import PreProcessingDef
from joda_al.utils.logging_utils.log_writers import gl_warning



def datasetFactory(preprocessing, in_memory, data_subpath, label_subpath, files_extractor, timestamp_extractor, label_converter):
    if in_memory:
        func=lambda path: PanopticProcessedSegmentationDatasetMem(path,data_subpath, label_subpath,
                                                         files_extractor, timestamp_extractor, label_converter,preprocessing)
    else:
        func=lambda path: PanopticProcessedSegmentationDataset(path,data_subpath, label_subpath,
                                                         files_extractor, timestamp_extractor, label_converter,preprocessing)
    return func


class PanopticSegmentationDataset(Dataset):
    _mean = None
    _std = None
    transform=None
    transforms =None
    def __init__(self, path: Union[List[str],str], input_subpath:str, label_subpath:str, files_extractor: Callable ,time_stamp_extractor:Optional[Callable], label_to_categorical: Callable, transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None, *args,**kwargs):
        """
        Load Kitti dataset from the given path, using path as local path and path_server for the server path.
        The server bool switches between the two paths.
        :param transforms: Dataset transformations - resize of images for classification
        :param path: path for the session
        """
        print(PanopticSegmentationDataset)
        # load all image files, sorting them to
        # ensure that they are aligned
        super().__init__(*args, **kwargs)
        if time_stamp_extractor is None:
            warn("Timestamps are not extracted")
            time_stamp_extractor= lambda x, y: ([],[])
        if self.transform is not None and transform is None:
            pass
        elif self.transform is not None and transform is not None:
            self.transform+=transform
        else:
            self.transform = transform
        if self.transforms is not None and transforms is None:
            pass
        elif self.transforms is not None and transforms is not None:
            self.transforms+=transforms
        else:
            self.transforms = transforms
        image_path, labels, dataset_timestamps, dataset_frame_ids, dataset_indicies = self.load_data(path, input_subpath, label_subpath, files_extractor, time_stamp_extractor)
        self.targets=labels
        self.dataset_indicies=dataset_indicies
        self.imgs = image_path
        self.frame_ids=np.array(dataset_frame_ids)
        self.timestamps=np.array(dataset_timestamps)
        self.label_to_categorical=label_to_categorical
        assert len(image_path) == len(labels)

    @staticmethod
    def load_data(paths: Union[List[str],str], input_subpath, label_subpath, files_extractor, time_stamp_extractor):
        """
        This function loads the data from the Kitti folder structure
        :param path: Path of the csv file of the images
        :return: Images and Class labels
        """
        if type(paths) == str:
            paths = [paths]
        dataset_img_paths = []
        dataset_label_paths = []
        dataset_indicies = []
        dataset_frame_ids = []
        dataset_timestamps = []
        for path in paths:
            input_path = os.path.join(path, input_subpath)
            label_path = os.path.join(path, label_subpath)
            session_img_paths, session_label_paths = files_extractor(input_path, label_path)
            # session_img_paths = sorted([os.path.join(input_path, img_name) for img_name in list(filter(lambda p: (".jpg" in p or ".png" in p), os.listdir(input_path)))])
            # session_label_paths = sorted([os.path.join(label_path, img_name) for img_name in
            #                             list(filter(lambda p: (".npy" in p), os.listdir(label_path)))])
            dataset_indicies.append([i for i in range(len(dataset_img_paths), len(dataset_img_paths)+len(session_img_paths))])
            sessions_frame_ids, session_timestamps = time_stamp_extractor(session_img_paths, session_label_paths)
            dataset_timestamps += session_timestamps
            dataset_frame_ids += sessions_frame_ids
            dataset_img_paths += session_img_paths
            dataset_label_paths += session_label_paths

            # Add indices before added new labels to reuse old length

            if len(dataset_img_paths) != len(dataset_label_paths):
                gl_warning(f"path: {path} labeled incorrect labels {len(dataset_img_paths)}, images {len(session_img_paths)}")

        return dataset_img_paths, dataset_label_paths, dataset_timestamps, dataset_frame_ids, dataset_indicies

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        # img_tensor = transforms.ToTensor()(img)
        targets = self.label_to_categorical(self.targets[idx])
        #self.to_onehot(targets)
        # targets_tensor = torch.squeeze(transforms.ToTensor()(targets))

        result = {
            "image": img,
            # "info": self.infos[index],
            **targets,
        }

        return result
        #return img_tensor, torch.tensor(targets,dtype=torch.long)

    def get_weights(self,idxs):
        """
        Calculate occurrence of the classes in the dataset for class weighted loss function
        :param idxs: list of indices for calculation
        :return: classwise weights
        """
        # TODO: this is broken
        labels=[self.targets[idx] for idx in idxs]
        occurences=Counter(labels)
        sum_weigths = sum(occurences.values())
        weights = [sum_weigths/v for k, v in sorted(occurences.items())]
        return torch.tensor(weights)/np.linalg.norm(weights)

    def __len__(self):
        return len(self.imgs)
    
    def set_mean_and_std(self,mean,std):
        self.mean=mean
        self.std=std

    def cal_mean_and_std(self,train_idx):
        imgs=[]
        for img_idx in train_idx:
            imgs.append(np.array(Image.open(self.imgs[img_idx]).convert("RGB")))
        image_array = np.array(imgs)
        self.mean=get_mean(image_array)
        self.std=get_std(image_array)
        return self.mean, self.std



class PanopticSegmentationDatasetMem(PanopticSegmentationDataset):

    def __init__(self, path:Union[List[str],str], input_subpath:str, label_subpath:str, files_extractor: Callable, time_stamp_extractor:Callable, label_to_categorical: Callable, *args,**kwargs):
        print(PanopticSegmentationDatasetMem)
        super().__init__(path, input_subpath, label_subpath, files_extractor, time_stamp_extractor, label_to_categorical, *args, **kwargs)
        self.imgs_array = self.load_images_parallel(self.imgs)
        # self.targets_list=np.array([label_to_categorical(path) for path in self.targets])
        self.targets_list = [label_to_categorical(path) for path in self.targets]  # Here path: image path (not folder path)

    @staticmethod
    def load_images_parallel(img_path_list,workers=8):
        try:
            #import ray.util.multiprocessing as mp
            import multiprocessing as mp
        except ImportError:
            import multiprocessing as mp
        with mp.Pool(workers) as p:
            imgs=p.map(load_parallel, img_path_list)
        return imgs # np.array(list(imgs))

    # @staticmethod
    # def load_images(img_path_list):
    #     imgs=[]
    #     for img_path in img_path_list:
    #         img = Image.open(img_path).convert("RGB")
    #         imgs.append(img)   # imgs.append(np.asarray(img).copy())
    #     return imgs # np.array(imgs)

    def cal_mean_and_std(self,train_idx):
        self.mean=get_mean(self.imgs_array, train_idx)
        self.std=get_std(self.imgs_array,train_idx)
        return self.mean, self.std

    def get_weights(self,idxs,reduction=2):
        """
        Calculate occurrence of the classes in the dataset for class weighted loss function
        :param idxs: list of indices for calculation
        :return: classwise weights
        """
        
        # labels=np.array([self.targets_list[idx] for idx in idxs])
        # values, counts = np.unique(np.array(labels),return_counts=True)
        # sum_weigths = np.sum(counts)
        # weights = np.array([sum_weigths/v for k, v in zip( values, counts )],dtype=np.float32)
        # weights**=(1/reduction)
        # weights/=np.min(weights)
        # return torch.tensor(weights,dtype=torch.float32)
        return False # TODO: fix the get_weights() when needed

    def __getitem__(self, idx):
        image_tensor=self.imgs_array[idx]
        # if self.transform is not None:
        #     img_tensor = self.transform(image_tensor)
        targets = self.targets_list[idx]
        # return img_tensor, torch.tensor(targets,dtype=torch.long)
        
        result = {
            "image": image_tensor,
            # "info": self.infos[index],
            **targets,
        }
        return result


def load_parallel(x):
    return Image.open(x).convert("RGB") # np.asarray(Image.open(x).convert("RGB"))


class PanopticProcessedSegmentationDatasetMem(PanopticSegmentationDatasetMem, DatasetPreprocessing):
    def __init__(self, path: Union[List[str], str], input_subpath: str, label_subpath: str, files_extractor: Callable,
                 time_stamp_extractor: Optional[Callable], label_to_categorical: Callable, preprocessing: PreProcessingDef,
                 *args, **kwargs):
        super(PanopticProcessedSegmentationDatasetMem, self).__init__(path, input_subpath, label_subpath, files_extractor, time_stamp_extractor,
                         label_to_categorical, *args, **kwargs,preprocessing=preprocessing)


class PanopticProcessedSegmentationDataset(PanopticSegmentationDataset, DatasetPreprocessing):
    def __init__(self, path: Union[List[str], str], input_subpath: str, label_subpath: str, files_extractor: Callable,
                 time_stamp_extractor: Optional[Callable], label_to_categorical: Callable, preprocessing: PreProcessingDef,
                 *args, **kwargs):
        super(PanopticProcessedSegmentationDataset, self).__init__(path, input_subpath, label_subpath, files_extractor, time_stamp_extractor,
                         label_to_categorical, *args, **kwargs,preprocessing=preprocessing)

def to_onehot(targets):
    return np.stack([np.eye(10)[tar] for tar in targets], axis=0)


def to_onehot_torch(targets,num_classes):
    return torch.stack([torch.stack([torch.eye(num_classes)[tar].T for tar in target], dim=1) for target in targets],dim=0)
