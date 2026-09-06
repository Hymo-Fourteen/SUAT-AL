import itertools
import os

from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter
from joda_al.data_loaders.dataset_registry import GTAV_LOCAL_NORM_UNI, GTAV_NORM, \
    GTAV_EQUALIZED, MEDIUM_IDENTIFIER
from joda_al.data_loaders.classification.stream_data_loader import A2D2_Dataset, A2D2_Dataset_Mem, \
    A2D2_Dataset_Mem_Norm, \
    A2D2_Dataset_Mem_Eq, A2D2_Dataset_Mem_LNorm

LABELS={"urban": 1, "highway": 2, "suburban": 6, "country_road": 4, "gravel_road":5, "residential": 3}


def labels_urban_highway_road_construction(labels):
    labels_u_h_r_c = {"urban": 1, "highway": 2, "suburban": 1, "country_road": 3, "gravel_road": 4, "residential": 1}
    return {k: labels_u_h_r_c[v]-1 for k,v in labels.items()}


def load_gta_v_stream(data_path,order,type="GTAVS"):

    dataset_subpath = "GTA-V/GTA-V-Streets"

    # Classification setup on reduced set.
    train_sessions = ["Route6"]
    unlabeled_sessions = [
        "Route2", "Route4", "Route5", "Route7"
    ]
    val_sessions =[
        "Route1", #Counter({'urban': 577, 'road': 244, 'construction': 72}) x
    ]
    test_sessions =[
        "Route3", #Counter({'highway': 1186, 'road': 67, 'urban': 30}) f
    ]
    if order == 1:
        unlabeled_sessions = unlabeled_sessions
    elif order == 2:
        unlabeled_sessions = unlabeled_sessions[::-1]
    elif order == 3:
        unlabeled_sessions = unlabeled_sessions[::2]+unlabeled_sessions[1::2]
    elif order == 4:
        unlabeled_sessions = [unlabeled_sessions[i-1] for i in [4, 5, 9, 2, 1, 7, 8, 6, 3]]
    elif order == 5:
        unlabeled_sessions = [unlabeled_sessions[i-1] for i in [2, 7, 3, 8, 9, 6, 5, 1, 4]]
    elif order == 6:
        unlabeled_sessions = [unlabeled_sessions[i-1] for i in [8, 7, 3, 4, 6, 5, 1, 9, 2]]
    else:
        raise NotImplementedError()
    if type in ["GTAVSm","GTAVS"]:
        ds_constructor=A2D2_Dataset_Mem
    elif type in GTAV_NORM:
        ds_constructor = A2D2_Dataset_Mem_Norm
    elif type in GTAV_EQUALIZED:
        ds_constructor = A2D2_Dataset_Mem_Eq
    elif type in GTAV_LOCAL_NORM_UNI:
        ds_constructor = A2D2_Dataset_Mem_LNorm
    else:
        raise NotImplementedError()
    labels_pre = labels_urban_highway_road_construction
    training_set = ds_constructor(labels_pre, [os.path.join(data_path, dataset_subpath, session) for session in train_sessions+unlabeled_sessions],label_file="labels.json")
    train_idx = training_set.dataset_indicies[0:1]
    pool_idx = training_set.dataset_indicies[1:]
    validation_set = ds_constructor(labels_pre, [os.path.join(data_path, dataset_subpath, session) for session in val_sessions],label_file="labels.json")
    test_set = ds_constructor(labels_pre, [os.path.join(data_path, dataset_subpath, session) for session in test_sessions],label_file="labels.json")
    return training_set, train_idx, pool_idx, validation_set, test_set, GTAVsScenarioConverter(type)


def get_feature_size(dataset_name):
    if MEDIUM_IDENTIFIER in dataset_name:
        feature_sizes_vgg = [(int(72 / 2 ** i), int(128 / 2 ** i)) for i in range(2, 6)]
        feature_size_resnet_torch=feature_sizes_vgg # [(18, 30), (9, 15), (4, 7), (2, 3)]
        #[(72, 120), (36, 60), (18, 30), (9, 15)]
        feature_size_no_max=[(int(72 / 2 ** i), int(128 / 2 ** i)) for i in range(1, 5)]
    else:
        feature_sizes_vgg = [(int(144 / 2 ** i), int(256 / 2 ** i)) for i in range(2, 6)]
        feature_size_no_max= [(int(144 / 2 ** i), int(256 / 2 ** i)) for i in range(1, 5)]
    return feature_size_no_max, feature_sizes_vgg, feature_sizes_vgg


class GTAVsScenarioConverter(DatasetScenarioConverter):

    def __init__(self,type):
        self.type=type

    def convert_task_to_multi_stream_dataset(self,training_pool, label_idx, unlabeled_idcs, validation_set, test_set):
        if "2" in self.type:
            unlabeled_idcs = [unlabeled_idcs[0:2], unlabeled_idcs[2:4]]
            return training_pool, label_idx, unlabeled_idcs, validation_set, test_set
        elif "3" in self.type:
            return DatasetScenarioConverter.convert_task_to_multi_stream_dataset_alternating(training_pool, label_idx, unlabeled_idcs, validation_set, test_set)
        else:
            return DatasetScenarioConverter.convert_task_to_multi_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set)


