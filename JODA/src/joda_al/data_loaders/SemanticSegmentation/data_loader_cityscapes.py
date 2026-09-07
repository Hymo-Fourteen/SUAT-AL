import itertools
import json
import os
from collections import namedtuple
from typing import Dict, Any, Union
import numpy as np
from PIL import Image

from joda_al.data_loaders.SemanticSegmentation.segmentation_dataset import datasetFactory
from joda_al.data_loaders.SemanticSegmentation.PanopticSegmentationDataset import datasetFactory as panopticDatasetFactory

from joda_al.data_loaders.dataset_registry import CITYSCAPES_LOCAL_NORM_UNI, \
    CITYSCAPES_EQUALIZED, CITYSCAPES_NORM, CITYSCAPES_PRE_NORM
from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter
from joda_al.defintions import PreProcessingDef
from joda_al.defintions import Engine
from joda_al.Config.config import load_experiment_config

"""`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

Args:
    root (string): Root directory of dataset where directory ``leftImg8bit``
        and ``gtFine`` or ``gtCoarse`` are located.
    split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
        otherwise ``train``, ``train_extra`` or ``val``
    mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
    target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
        or ``color``. Can also be a list to output a tuple with all specified target types.
    transform (callable, optional): A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    transforms (callable, optional): A function/transform that takes input sample and its target as entry
        and returns a transformed version.

Examples:

    Get semantic segmentation target

    .. code-block:: python

        dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                             target_type='semantic')

        img, smnt = dataset[0]

    Get multiple targets

    .. code-block:: python

        dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                             target_type=['instance', 'color', 'polygon'])

        img, (inst, col, poly) = dataset[0]

    Validate on the "coarse" set

    .. code-block:: python

        dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                             target_type='semantic')

        img, smnt = dataset[0]
"""

# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = namedtuple(
    "CityscapesClass",
    ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
)

classes = [
    CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]
id_to_color = [tuple(map(lambda x: x/255,c.color)) for c in classes ]
train_id_to_color = [tuple(map(lambda x: x/255,c.color)) for c in classes if (c.train_id != -1 and c.train_id != 255)]
id_to_train_id = np.array([c.train_id for c in classes])
ignored_train_id = [c.train_id for c in classes if c.ignore_in_eval == True]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as file:
        data = json.load(file)
    return data


def _get_target_suffix(mode: str, target_type: str) -> str:
    if target_type == "instance":
        return f"{mode}_instanceIds.png"
    elif target_type == "semantic":
        return f"{mode}_labelIds.png"
    elif target_type == "color":
        return f"{mode}_color.png"
    else:
        return f"{mode}_polygons.json"

def create_mapping(value_dict : Union[Dict,np.ndarray]):
    if isinstance(value_dict,np.ndarray):
        value_dict={i:v if v not in [255,-1] else np.int64(-100) for i,v in enumerate(value_dict)}
    def map_to_trainings_values(label_ar):
        return np.vectorize(value_dict.get)(label_ar)
    return map_to_trainings_values

def get_label_converter(mode="gtFine",target_type="semantic",mapping=None):
    def label_converter(file_name):
        label_name="{}_{}".format(file_name.split("_leftImg8bit")[0], _get_target_suffix(mode, target_type))
        label=Image.open(label_name)
        label_ar=np.array(label)
        if mapping is not None:
            label_ar= mapping(label_ar)
        return label_ar

    return label_converter

def get_label_converter_types(mode="gtFine",target_type="semantic",mapping=None):
    def label_converter(file_name):
        labels={}
        if isinstance(target_type,list):  # TODO: write the else condition too (if not type:list)
            for target_t in target_type:
                label_name="{}_{}".format(file_name.split("_leftImg8bit")[0], _get_target_suffix(mode, target_t))
                label=Image.open(label_name)
                # label_ar=np.array(label)
                # if mapping is not None and target_t=="semantic":
                    # label_ar= mapping(label_ar)
                labels[target_t] = label # label_ar
        return labels

    return label_converter


def map_target_id_to_train_id(target):
    return id_to_train_id[np.array(target)]


def files_getter(image_path, label_path):
    session_img_paths = sorted([os.path.join(image_path, img_name) for img_name in
                                list(filter(lambda p: (".png" in p), os.listdir(image_path)))])
    session_label_paths=[p.replace("/leftImg8bit","/gtFine") for p in session_img_paths]
    return session_img_paths, session_label_paths


def load_cityscapes_stream(data_path, order, type="",):
    "{0: 398, 1: 577, 2: 215} 1090"
    # Classification setup on reduced set.
    #[construction, construction_vulnerable, non_crit, vulnerable]
    train_sessions = [
        "aachen",  # [200  64 706 182]
        "bochum",  # [0 1 52 44]
        # "dresden",  # [ 33  41 181 167]
    ]
    # aachen, bochum, bremen, cologne, darmstadt, dusseldorf, erfurt, hamburg, \
    # hanover,\
    #     jena, krefeld, monchengladbach, strasbourg, stuttgart, tubingen, ulm, weimar, zurich
    unlabeled_sessions = [
        "bremen",  # [ 42  30 471 187]
        "cologne",  # [ 42  24 747 388]
        "darmstadt",  # [ 33  41 181 167]
        "dusseldorf",  # [ 57  31 620 404]
        "erfurt",  # [ 47   9 644 155]
        "hamburg",  # [ 19  32 226 168]
        "hanover",  # [ 57  18 978 451]
        "jena",
        "krefeld",
        "strasbourg",
        "stuttgart",
        "tubingen",
    ]
    val_sessions = [
        "ulm",
        "weimar",
        "zurich",
        "monchengladbach",
    ]
    test_sessions = [
        "frankfurt",
        "lindau",
        "munster",
    ]
    # test_sessions = [
    #     "berlin",
    #     "bielefeld",
    #     "bonn",
    #     "leverkusen",
    #     "mainz",
    #     "munich"
    # ]
    if "m" in type:
        dataset_subpath = "cityscapes/leftImg8bit_trainvaltest/leftImg8bit_small"
        mode = "gtFine"
        in_memory = True
    elif "l" in type:
        dataset_subpath = "cityscapes/leftImg8bit_trainvaltest/leftImg8bit_medium"
        mode = "gtFine"
        in_memory = True
    else:
        dataset_subpath = "cityscapes/leftImg8bit_trainvaltest/leftImg8bit"
        mode = "gtFine"
        in_memory = False
    if order==1:
        unlabeled_sessions=unlabeled_sessions
    elif order==2:
        unlabeled_sessions = unlabeled_sessions[::-1]
    elif order == 3:
        unlabeled_sessions = unlabeled_sessions[::2]+unlabeled_sessions[1::2]
    else:
        raise NotImplementedError()
    if type in CITYSCAPES_EQUALIZED:
        preprocessing = PreProcessingDef.equalized
    elif type in CITYSCAPES_LOCAL_NORM_UNI:
        preprocessing = PreProcessingDef.local_maxed
    elif type in CITYSCAPES_PRE_NORM:
        preprocessing = PreProcessingDef.pretrained_norm
    elif type in CITYSCAPES_NORM:
        preprocessing = PreProcessingDef.norm
    else:
        preprocessing = PreProcessingDef.none

    timestamp_extractor = None


    ds_constructor = datasetFactory(preprocessing, in_memory, "", "categorical_reduced_label/cam_front_center",
                                                         files_getter, timestamp_extractor, get_label_converter(mode=mode,mapping=create_mapping(id_to_train_id)))




    training_set = ds_constructor(
        [os.path.join(data_path, dataset_subpath, "train", session) for session in train_sessions + unlabeled_sessions])
    train_idx = training_set.dataset_indicies[0:len(train_sessions)]
    pool_idx = training_set.dataset_indicies[len(train_sessions):]
    validation_set = ds_constructor([os.path.join(data_path, dataset_subpath, "train", session) for session in val_sessions])
    test_set = ds_constructor([os.path.join(data_path, dataset_subpath, "val", session) for session in test_sessions])
    return training_set, train_idx, pool_idx, validation_set, test_set, [], CityscapesScenarioConverter(type)

class CityscapesScenarioConverter(DatasetScenarioConverter):

    def __init__(self,type):
        self.type=type

    def convert_task_to_multi_stream_dataset(self, training_pool, label_idx, unlabeled_idcs, validation_set, test_set):
        if "2" not in self.type:
            unlabeled_idcs_batch=[unlabeled_idcs[0:3],unlabeled_idcs[3:6],unlabeled_idcs[6:9],unlabeled_idcs[9:12]]
            return training_pool, label_idx, unlabeled_idcs_batch, validation_set, test_set
        else:
            num_sub_streams = 2
            sub_streams = []
            for un_lab in unlabeled_idcs:
                leftover = len(un_lab) % num_sub_streams
                for i in range(num_sub_streams):
                    sub_streams.append(
                        [un_lab[j] for j in range(i, len(un_lab) + i + 1 - num_sub_streams, num_sub_streams)])
                if leftover > 0:
                    sub_streams[-1].extend(un_lab[-leftover:])
            unlabeled_idcs = sub_streams
            unlabeled_idcs_batch = [unlabeled_idcs[0:6], unlabeled_idcs[6:12], unlabeled_idcs[12:18], unlabeled_idcs[18:24]]
            return training_pool, label_idx, unlabeled_idcs_batch, validation_set, test_set


def get_feature_size(dataset_name):
    if "m" in dataset_name:
        width=512
        heigth=256
    else:
        width=2048
        heigth=1024
    feature_sizes = {}
    feature_sizes["resnet"] = [(256, 512), (128, 256), (128, 256), (128, 256)]
    feature_sizes["mobilenet"] = [(128, 256), (64, 128), (64, 128), (64, 128)]
    feature_sizes["resnet34"] = [(int(heigth / 2 ** i), int(width / 2 ** i)) for i in range(2, 6)]
    # feature_sizes["mobilenet"] = [(150, 240), (75, 120), (38, 60), (19, 30), (19, 30), (19, 30)]
    feature_sizes["resnet50"]=[(int(heigth / 2 ** i), int(width / 2 ** i)) for i in range(2, 4)]
    feature_sizes["resnet50"]+=[feature_sizes["resnet50"][-1]]*2
    return feature_sizes