import json
import math
import os

from PIL import Image
import numpy as np

from joda_al.data_loaders.SemanticSegmentation.segmentation_dataset import datasetFactory
from joda_al.data_loaders.dataset_registry import A2D2_LOCAL_NORM_UNI, \
    A2D2_EQUALIZED, A2D2_PRE_NORM, A2D2_NORM
from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter
from joda_al.defintions import PreProcessingDef

labels = [
    "ignored",
    "nature",
    "buildings",
    "traffic_guide",
    "non_drivable",
    "ego_car", # 5
    "bicycle",
    "pedestrian", # 7
    "small_moiving_objects", # 8
    "moving_medium_objects", # 9
    "moving_big_objects",
    "sky",
    "street_areas",
    "guiding",
    "lane_markings",
    "lane_lines",
    "sidewalks",
    "obstacles",
]

#Full labels:


def load_A2D2_stream(data_path, order, type="A2D2"):
    """
    1) AEV -> garbenheim
    2) AEV -> garbenheim
    3) Munich 700 tunnel ?
    4) MUC
    5) MUC - mittlerer Ring keine Bundestraße
    6) MUC
    7) Augsburg oder IN Straßeneisenbahn (annomalie)?
    8) IN? img 183 transport of construction equipment ~ 800 highwayentrence inside urban
    9) reallly high framerate, many similar images
    10) high framerate but in the end exit the city? with federal road?
    11) only crossing
    12) highway
    13) highway with exit
    14) überland fahrt some villages difficult urban/suburban labeling
    15) country road, suburbs and highway afterwards ends with federal road
    16) country road and highway change and rastplatz in the end
    17) suburban and country road
    18) rainy in the end block sensors
    """
    if "L" in type:
        dataset_subpath = "A2D2/camera_lidar_semantic_medium_full"
        in_meomory=True
        camera_path = "camera_bi"
    elif "F" in type:
        dataset_subpath = "A2D2/camera_lidar_semantic"
        in_meomory = False
        camera_path = "camera"
    elif "B" in type:
        dataset_subpath = "A2D2/camera_lidar_semantic_bboxes"
        in_meomory = False
        camera_path = "camera"
    elif "s" in type:
        dataset_subpath = "A2D2/camera_lidar_semantic_small"
        in_meomory = True
        camera_path="camera_bi"
    else:
        dataset_subpath = "A2D2/camera_lidar_semantic_medium"
        in_meomory = True
        camera_path="camera_bi"
    # Classification setup on reduced set.
    train_sessions = ["20181107_132730",  # Counter({'urban': 588, 'construction': 149, 'road': 75}) x
                      "20181108_091945",  # Counter({'highway': 780, 'road': 48, 'urban': 33, 'construction': 1}) f
                      ]
    unlabeled_sessions = [
        "20181107_133258",  # Counter({'road': 117}) x
        "20181108_084007",  # Counter({'highway': 643}) x
        "20180807_145028",  # Counter({'urban': 742, 'road': 110, 'construction': 90}) x
        "20180810_142822",  # Counter({'urban': 499, 'construction': 35, 'road': 29}) x
        "20180925_135056",  # Counter({'urban': 610, 'construction': 82}) x
        "20181008_095521",  # Counter({'urban': 608, 'construction': 119}) x
        "20181107_132300",  # Counter({'urban': 405, 'road': 145, 'construction': 21}) f
        "20181204_154421",  # Counter({'road': 92, 'urban': 55, 'construction': 4}) f
        "20181204_170238",  # Counter({'road': 99, 'urban': 13}) f
    ]
    val_sessions = [
        "20180925_101535",  # Counter({'urban': 865, 'construction': 195}) x
        "20181016_125231",  # Counter({'urban': 577, 'road': 244, 'construction': 72}) x
        "20181204_135952"  # Counter({'highway': 563, 'road': 59, 'construction': 2})" f
    ]
    test_sessions = [
        "20180925_124435",  # Counter({'urban': 820, 'construction': 156}) x
        "20181108_123750",  # Counter({'highway': 1186, 'road': 67, 'urban': 30}) f
        "20181108_103155"  # Counter({'urban': 273, 'road': 232, 'construction': 12}) f
    ]
    # train_sessions = ["20180807_145028"]
    # val_sessions = ["20180807_145028"]
    # test_sessions = ["20180807_145028"]
    # unlabeled_sessions = ["20180807_145028"]
    if order == 1:
        unlabeled_sessions = unlabeled_sessions
    elif order == 2:
        unlabeled_sessions = unlabeled_sessions[::-1]
    elif order == 3:
        unlabeled_sessions = unlabeled_sessions[::2] + unlabeled_sessions[1::2]
    elif order == 4:
        unlabeled_sessions = [unlabeled_sessions[i - 1] for i in [4, 5, 9, 2, 1, 7, 8, 6, 3]]
    elif order == 5:
        unlabeled_sessions = [unlabeled_sessions[i - 1] for i in [2, 7, 3, 8, 9, 6, 5, 1, 4]]
    elif order == 6:
        unlabeled_sessions = [unlabeled_sessions[i - 1] for i in [8, 7, 3, 4, 6, 5, 1, 9, 2]]
    else:
        raise NotImplementedError()
    # label_converter = lambda label_path: np.load(label_path, allow_pickle=True)  # for numpy
    label_converter = lambda label_path: np.asarray(Image.open(label_path)) # Cat image
    # label_converter=build_categorical_image_converter(os.path.join(data_path,dataset_subpath)) # for cat non reduced
    if type in A2D2_EQUALIZED:
        preprocessing = PreProcessingDef.equalized
    elif type in A2D2_LOCAL_NORM_UNI:
        preprocessing = PreProcessingDef.local_maxed
    elif type in A2D2_PRE_NORM:
        preprocessing = PreProcessingDef.pretrained_norm
    elif type in A2D2_NORM:
        preprocessing = PreProcessingDef.norm
    else:
        preprocessing = PreProcessingDef.none


    files_extractor=files_getter_cat_image
    ds_constructor = datasetFactory(preprocessing, in_meomory, f"{camera_path}/cam_front_center", "categorical_reduced_label/cam_front_center",
                                                         files_extractor, timestamp_extractor, label_converter)
    training_set = ds_constructor(
        [os.path.join(data_path, dataset_subpath, session) for session in train_sessions + unlabeled_sessions])
    train_idx = training_set.dataset_indicies[0:2]
    pool_idx = training_set.dataset_indicies[2:]
    validation_set = ds_constructor([os.path.join(data_path, dataset_subpath, session) for session in val_sessions])
    test_set = ds_constructor([os.path.join(data_path, dataset_subpath, session) for session in test_sessions])
    ignored_classes = [0,5,17]
    return training_set, train_idx, pool_idx, validation_set, test_set, ignored_classes, A2D2ScenarioConverter()


class A2D2ScenarioConverter(DatasetScenarioConverter):
    @staticmethod
    def convert_task_to_multi_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set):
        unlabeled_idcs=[unlabeled_idcs[0:3],unlabeled_idcs[3:6],unlabeled_idcs[6:9]]
        return training_pool, label_idx, unlabeled_idcs, validation_set, test_set



def create_label_to_categorical(dataset_path):
    # class colour mapping
    file = os.path.join(dataset_path, "class_list.json")
    with open(file, "r") as f:
        class_colour_mapping = json.load(f)

    # class group mapping
    file = os.path.join(dataset_path, "class_group_mapping.json")
    with open(file, "r") as f:
        groups_class_mapping = json.load(f)

    group_class_name_to_id = {k: i for i, k in enumerate(groups_class_mapping.keys())}
    class_groups_mapping = {
        category_id: group_class_name_to_id[class_group]
        for class_group, classes in groups_class_mapping.items()
        for category_id in classes
    }
    colour_to_class_group_id = {
        k: class_groups_mapping[v] for k, v in class_colour_mapping.items()
    }
    reduced_mapping = {k.replace("#", ""): v for k, v in colour_to_class_group_id.items()}

    hexa = np.vectorize("{:02x}".format)

    def label_to_categorical(label_path):
        img = Image.open(label_path).convert("RGB")
        image_1dhexa = np.sum(hexa(img).astype(object), axis=-1)
        categorical = np.vectorize(reduced_mapping.get)(image_1dhexa).astype(np.uint8)
        return categorical

    return label_to_categorical


def build_categorical_image_converter(dataset_path):
    # class colour mapping
    file = os.path.join(dataset_path, "class_list.json")
    with open(file, "r") as f:
        class_colour_mapping = json.load(f)
    class_colour_mapping={v: idx for idx,(k, v) in enumerate(class_colour_mapping.items())}
    # class group mapping
    file = os.path.join(dataset_path, "class_group_mapping.json")
    with open(file, "r") as f:
        groups_class_mapping = json.load(f)

    group_class_name_to_id = {k: i for i, k in enumerate(groups_class_mapping.keys())}
    class_groups_mapping = {
        group_class_name_to_id[class_group]: [class_colour_mapping[c] for c in classes]
        for class_group, classes in groups_class_mapping.items()
        }

    def categorical_image_converter(label_name):
        label = Image.open(label_name)
        onehot_labels=np_one_hot(np.array(label),len(class_colour_mapping))
        reduced_labels=np.zeros_like(label)
        for i, class_groups_map in class_groups_mapping.items():
            reduced_labels+=np.sum(onehot_labels[:,:,class_groups_map],axis=-1)*i
        return reduced_labels
    return categorical_image_converter


def files_getter(image_path,label_path):
    session_img_paths = sorted([os.path.join(image_path, img_name) for img_name in
                                list(filter(lambda p: (".jpg" in p or ".png" in p), os.listdir(image_path)))])
    session_label_paths = sorted([os.path.join(label_path, img_name) for img_name in
                                  list(filter(lambda p: (".npy" in p), os.listdir(label_path)))])
    return session_img_paths,session_label_paths


def files_getter_cat_image(image_path,label_path):
    session_img_paths = sorted([os.path.join(image_path, img_name) for img_name in
                                list(filter(lambda p: (".jpg" in p or ".png" in p), os.listdir(image_path)))])
    session_label_paths = sorted([os.path.join(label_path, img_name) for img_name in
                                  list(filter(lambda p: (".png" in p), os.listdir(label_path)))])
    return session_img_paths,session_label_paths


def get_feature_size(dataset_name):
    if "B" in dataset_name or "F" in dataset_name:
        width=1920
        heigth=1208
    else:
        width=640
        heigth=400
    feature_sizes = {}
    feature_sizes["resnet"] = [(75, 120), (38, 60), (38, 60), (38, 60)]
    # feature_sizes["resnet34"] = [(302, 480), (151, 240), (76, 120), (38, 60)]
    feature_sizes["resnet34"] = [(int(normal_round(heigth / 2 ** i)), int(normal_round(width / 2 ** i))) for i in range(2, 6)]
    # feature_sizes["mobilenet"] = [(150, 240), (75, 120), (38, 60), (19, 30), (19, 30), (19, 30)]
    feature_sizes["resnet50"]=[(int(normal_round(heigth / 2 ** i)), int(normal_round(width / 2 ** i))) for i in range(2, 4)]
    feature_sizes["resnet50"] += [feature_sizes["resnet50"][-1]] * 2
    feature_sizes["mobilenet"] = [(150, 240), (75, 120), (38, 60), (19, 30), (19, 30), (19, 30)]
    return feature_sizes


def timestamp_extractor(image_paths, label_paths):
    dataset_frame_ids = [int(k.split("_")[-1].split(".png")[0]) for k in image_paths]
    dataset_timestamps = []
    for img in image_paths:
        with open(img.replace(".png", ".json"), "r") as metadata:
            dataset_timestamps.append(float(json.load(metadata)["cam_tstamp"]))
    return dataset_frame_ids, dataset_timestamps


def normal_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals


def np_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return np.array(res.reshape(list(targets.shape)+[nb_classes]),dtype=np.uint8)
