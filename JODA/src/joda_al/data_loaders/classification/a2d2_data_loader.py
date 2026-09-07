import os

from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter
from joda_al.data_loaders.dataset_registry import A2D2_EQUALIZED, \
    A2D2_LOCAL_NORM_UNI, LARGE_IDENTIFIER, SMALL_IDENTIFIER, MEDIUM_IDENTIFIER, A2D2_NORM, A2D2_PRE_NORM
from joda_al.data_loaders.classification.stream_data_loader import A2D2_Dataset, A2D2_Dataset_Mem, \
    A2D2_Dataset_Mem_Norm, \
    A2D2_Dataset_Mem_Eq, A2D2_Dataset_Mem_LNorm, A2D2_Dataset_Pre_Norm

LABELS={"urban":1, "highway":2, "suburban":3, "country_road":4, "federal_road":5, "urban_construction_side":6, "highway_construction_side":7, "suburban_construction_side":8, "country_road_construction_side":9, "federal_road_construction_side":10, "urban_tunnel":11, "urban_tunnel_construction_side":12, "highway_tunnel":13, "highway_tunnel_construction_side":14, "high_density_urban":15, "high_density_urban_construction_side":16}


def labels_urban_highway_road_construction(labels):
    labels_u_h_r_c = {"urban": 1, "highway": 2, "suburban": 1, "country_road": 3, "federal_road": 3,
              "urban_construction_side": 4, "highway_construction_side": 4, "suburban_construction_side": 4,
              "country_road_construction_side": 4, "federal_road_construction_side": 4, "urban_tunnel": 1,
              "urban_tunnel_construction_side": 4, "highway_tunnel": 2, "highway_tunnel_construction_side": 4,
              "high_density_urban": 1, "high_density_urban_construction_side": 4}
    return {k: labels_u_h_r_c[v]-1 for k,v in labels.items()}


def load_A2D2_stream(data_path,order,type="A2D2"):
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
        dataset_subpath = "A2D2/classification_labels_full"
    else:
        dataset_subpath="A2D2/classification_labels"

    # Classification setup on reduced set.
    train_sessions = ["20181107_132730", #Counter({'urban': 588, 'construction': 149, 'road': 75}) x
                      "20181108_091945", #Counter({'highway': 780, 'road': 48, 'urban': 33, 'construction': 1}) f
                     ]
    unlabeled_sessions = [
        "20181107_133258", #Counter({'road': 117}) x   L 191
        "20181108_084007", #Counter({'highway': 643}) x L 1424
        "20180807_145028", #Counter({'urban': 742, 'road': 110, 'construction': 90}) x
        "20180810_142822", #Counter({'urban': 499, 'construction': 35, 'road': 29}) x
        "20180925_135056", #Counter({'urban': 610, 'construction': 82}) x
        "20181008_095521", #Counter({'urban': 608, 'construction': 119}) x
        "20181107_132300", #Counter({'urban': 405, 'road': 145, 'construction': 21}) f
        "20181204_154421", #Counter({'road': 92, 'urban': 55, 'construction': 4}) f
        "20181204_170238", #Counter({'road': 99, 'urban': 13}) f
    ]
    val_sessions =[
        "20180925_101535", #Counter({'urban': 865, 'construction': 195}) x
        "20181016_125231", #Counter({'urban': 577, 'road': 244, 'construction': 72}) x
        "20181204_135952" #Counter({'highway': 563, 'road': 59, 'construction': 2})" f
    ]
    test_sessions =[
        "20180925_124435", #Counter({'urban': 820, 'construction': 156}) x
        "20181108_123750", #Counter({'highway': 1186, 'road': 67, 'urban': 30}) f
        "20181108_103155" #Counter({'urban': 273, 'road': 232, 'construction': 12}) f
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
    elif order == 11:
        val_session_extra = unlabeled_sessions[-1]
        test_session_extra = unlabeled_sessions[-2]
        val_sessions.append(val_session_extra)
        test_sessions.append(test_session_extra)
        del unlabeled_sessions[-1]
        del unlabeled_sessions[-2]
        del unlabeled_sessions[0]
    elif order == 12:
        val_session_extra = unlabeled_sessions[-1]
        test_session_extra = unlabeled_sessions[-2]
        val_sessions.append(val_session_extra)
        test_sessions.append(test_session_extra)
        del unlabeled_sessions[-1]
        del unlabeled_sessions[-2]
        del unlabeled_sessions[0]
        unlabeled_sessions = unlabeled_sessions[::-1]
    elif order == 13:
        val_session_extra = unlabeled_sessions[-1]
        test_session_extra = unlabeled_sessions[-2]
        val_sessions.append(val_session_extra)
        test_sessions.append(test_session_extra)
        del unlabeled_sessions[-1]
        del unlabeled_sessions[-2]
        del unlabeled_sessions[0]
        unlabeled_sessions = unlabeled_sessions[::2] + unlabeled_sessions[1::2]
    else:
        raise NotImplementedError()
    if type == "A2D2":
        ds_constructor=A2D2_Dataset
    elif type in A2D2_NORM:
        ds_constructor = A2D2_Dataset_Mem_Norm
    elif type in A2D2_EQUALIZED:
        ds_constructor = A2D2_Dataset_Mem_Eq
    elif type in A2D2_LOCAL_NORM_UNI:
        ds_constructor = A2D2_Dataset_Mem_LNorm
    elif type in A2D2_PRE_NORM:
        ds_constructor = A2D2_Dataset_Pre_Norm
    else:
        # elif type in ["A2D2M", "A2D2R", "A2D2R2", "A2D2R3", 'A2D2l', 'A2D2m', 'A2D2lL', 'A2D2mL']:
        ds_constructor = A2D2_Dataset_Mem
    labels_pre = labels_urban_highway_road_construction
    training_set = ds_constructor(labels_pre, [os.path.join(data_path, dataset_subpath, session) for session in train_sessions+unlabeled_sessions])
    train_idx = training_set.dataset_indicies[0:len(train_sessions)]
    pool_idx = training_set.dataset_indicies[len(train_sessions):]
    validation_set = ds_constructor(labels_pre, [os.path.join(data_path, dataset_subpath, session) for session in val_sessions])
    test_set = ds_constructor(labels_pre, [os.path.join(data_path, dataset_subpath, session) for session in test_sessions])
    return training_set, train_idx, pool_idx, validation_set, test_set, DatasetScenarioConverter()
    # inital_set = [ A2D2_Dataset(labels_pre,os.path.join(data_path,session)) for session in train_sessions]
    # unlabeled_set = [A2D2_Dataset( labels_pre, os.path.join(data_path, session)) for session in unlabeled_sessions]
    # validation_set = [A2D2_Dataset( labels_pre, os.path.join(data_path, session)) for session in val_sessions]
    # test_set = [A2D2_Dataset( labels_pre, os.path.join(data_path, session)) for session in test_sessions]
    # return inital_set, unlabeled_set, validation_set, test_set

def get_sample_size(dataset_name):
    if LARGE_IDENTIFIER in dataset_name:  # in ["A2D2lN", 'A2D2l', 'A2D2lE', 'A2D2lU']:
        return (144, 240)
    elif MEDIUM_IDENTIFIER in dataset_name:  # dataset_name in ["A2D2mN", "A2D2m","A2D2mNR"]:
        return (72, 120)
    elif SMALL_IDENTIFIER in dataset_name:
        return (36,60)
    else:
        return (151, 240)


def get_feature_size(dataset_name):
    if LARGE_IDENTIFIER in dataset_name: # in ["A2D2lN", 'A2D2l', 'A2D2lE', 'A2D2lU']:
        feature_sizes_vgg = [(int(144 / 2 ** i), int(240 / 2 ** i)) for i in range(2, 6)]
        feature_size_no_max= [(int(144 / 2 ** i), int(240 / 2 ** i)) for i in range(1, 5)]
        feature_size_resnet_torch = feature_sizes_vgg
    elif MEDIUM_IDENTIFIER in dataset_name: #dataset_name in ["A2D2mN", "A2D2m","A2D2mNR"]:
        feature_sizes_vgg = [(int(72 / 2 ** i), int(120 / 2 ** i)) for i in range(2, 6)]
        feature_size_resnet_torch=feature_sizes_vgg # [(18, 30), (9, 15), (4, 7), (2, 3)]
        #[(72, 120), (36, 60), (18, 30), (9, 15)]
        feature_size_no_max=[(int(72 / 2 ** i), int(120 / 2 ** i)) for i in range(1, 5)]
    elif SMALL_IDENTIFIER in dataset_name:
        feature_sizes_vgg = [(int(36 / 2 ** i), int(60 / 2 ** i)) for i in range(2, 6)]
        feature_size_resnet_torch=feature_sizes_vgg # [(18, 30), (9, 15), (4, 7), (2, 3)]
        feature_size_resnet_torch[-1]=(2,2)
        #[(72, 120), (36, 60), (18, 30), (9, 15)]
        feature_size_no_max=[(int(36 / 2 ** i), int(60 / 2 ** i)) for i in range(1, 5)]
    else:
        feature_sizes_vgg = [(int(151 / 2 ** i), int(240 / 2 ** i)) for i in range(2, 6)]
        feature_size_no_max = [(int(151 / 2 ** i), int(240 / 2 ** i)) for i in range(1, 5)]
        feature_size_resnet_torch=feature_sizes_vgg
    return feature_size_no_max, feature_size_resnet_torch, feature_sizes_vgg

