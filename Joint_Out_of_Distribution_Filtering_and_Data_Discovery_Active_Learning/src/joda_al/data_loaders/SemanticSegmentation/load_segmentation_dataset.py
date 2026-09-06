import itertools

from matplotlib import colors
from torchvision.transforms import InterpolationMode
from joda_al.data_loaders.SemanticSegmentation.data_loader_a2d2_seg import load_A2D2_stream, get_feature_size as a2d2_features
from joda_al.data_loaders.SemanticSegmentation.data_loader_cityscapes import load_cityscapes_stream, \
    get_feature_size as city_features, id_to_color
from joda_al.data_loaders.dataset_lib import update_means
from joda_al.Pytorch_Extentions import transforms


def load_segsem_dataset(dataset, config, order=1):
    if "A2D2" in dataset["name"]:
        feature_sizes = set_feature_size(a2d2_features(dataset["name"]))

        color_names=["black","green","grey","lightgrey","pink","peru","lightsalmon","red","firebrick","darkred","sienna","skyblue","magenta","gold","lime","mediumseagreen","purple","teal"]
        color_list=[]
        for col in color_names:
            hex_color = colors.get_named_colors_mapping()[col]
            rgb_color = colors.to_rgb(hex_color)
            color_list.append(rgb_color)

        training_pool, label_idcx, unlabeled_idcs, validation_set, test_set, ignored_classes, factory = load_A2D2_stream(dataset["path"], order,
                                                                                               dataset["name"])
        dataset_config = {"num_classes": 18, "feature_sizes": feature_sizes, "encoding_dimension": 64,
                          "ignored_classes": ignored_classes, "color_list": color_list, "dataset_name":"A2D2"}
        label_idx = list(itertools.chain(*label_idcx))
        update_means(dataset["name"], label_idx, training_pool, validation_set, test_set)
    elif "cityscapes" in dataset["name"]:
        feature_sizes = set_feature_size(city_features(dataset["name"]))
        color_list=id_to_color
        training_pool, label_idcx, unlabeled_idcs, validation_set, test_set, ignored_classes, factory = load_cityscapes_stream(
            dataset["path"], order,
            dataset["name"])
        dataset_config = {"num_classes": 19, "feature_sizes": feature_sizes, "encoding_dimension": 240,
                          "ignored_classes": ignored_classes, "color_list":color_list, "dataset_name":"Cityscapes"}
        label_idx = list(itertools.chain(*label_idcx))
        update_means(dataset["name"], label_idx, training_pool, validation_set, test_set)
    else:
        raise NotImplementedError()
    dataset_config["name"] = dataset["name"]
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, factory


def set_feature_size(feature_size):
    feature_sizes = {
        "deeplabv3_resnet50": feature_size["resnet50"],
        "deeplabv3_resnet34": feature_size["resnet34"],
        "deeplabv3_mobilenet_v3": feature_size["mobilenet"],
        "deeplabv3plus_resnet50": feature_size["resnet50"],
        "deeplabv3plus_resnet34": feature_size["resnet34"],
        "deeplabv3plus_mobilenet_v3": feature_size["mobilenet"],
    }
    return feature_sizes


def get_training_transformations_semseg(dataset_name):
    trans_map = {
        "A2D2R": transforms.CombinedRandomHorizontalFlip(),
        "A2D2NR": transforms.CombinedRandomHorizontalFlip(),
        "A2D2BNR": transforms.CombinedCompose([transforms.CombinedRandomApply([transforms.CombinedRandomResizedCrop((400, 640),scale=(0.75, 1.0), ratio=(16 / 10, 16 / 10),interpolation=InterpolationMode.BILINEAR)]),transforms.CombinedRandomHorizontalFlip(p=0.3)]),
        "A2D2TNR": transforms.CombinedCompose([transforms.CombinedRandomApply([transforms.CombinedRandomResizedCrop(
            (400, 640), scale=(0.75, 1.0), ratio=(16 / 10, 16 / 10), interpolation=InterpolationMode.BILINEAR)]),
                                               transforms.CombinedRandomHorizontalFlip(p=0.3)]),

        "cityscapesNR": transforms.CombinedRandomCrop((768,768)),
        "cityscapesPR": transforms.CombinedCompose([transforms.CombinedRandomCrop((768, 768)),transforms.CombinedRandomHorizontalFlip()]),
        "cityscapesmNR": transforms.CombinedCompose(
            [transforms.CombinedRandomResizedCrop(
            (256, 512), scale=(0.75, 1.0), ratio=(18 / 10, 22 / 10), interpolation=InterpolationMode.BILINEAR), transforms.CombinedRandomHorizontalFlip()]),
        "cityscapesmNR2": transforms.CombinedCompose(
            [transforms.CombinedRandomResizedCrop(
                (256, 512), scale=(0.75, 1.0), ratio=(18 / 10, 22 / 10), interpolation=InterpolationMode.BILINEAR),
                transforms.CombinedRandomHorizontalFlip()]),
        "cityscapesmNr": transforms.CombinedCompose(
            [transforms.CombinedRandomResizedCrop(
                (256, 512), scale=(0.75, 1.25), ratio=(18 / 10, 22 / 10), interpolation=InterpolationMode.BILINEAR),
                transforms.CombinedRandomHorizontalFlip()]),
        "cityscapesmNr2": transforms.CombinedCompose(
            [transforms.CombinedRandomResizedCrop(
                (256, 512), scale=(0.75, 1.25), ratio=(18 / 10, 22 / 10), interpolation=InterpolationMode.BILINEAR),
                transforms.CombinedRandomHorizontalFlip()]),
        "cityscapeslNR": transforms.CombinedCompose(
            [transforms.CombinedRandomResizedCrop(
                (512, 1024), scale=(0.75, 1.0), ratio=(18 / 10, 22 / 10), interpolation=InterpolationMode.BILINEAR),
                transforms.CombinedRandomHorizontalFlip()]),
        "cityscapeslNR2": transforms.CombinedCompose(
            [transforms.CombinedRandomResizedCrop(
                (512, 1024), scale=(0.75, 1.0), ratio=(18 / 10, 22 / 10), interpolation=InterpolationMode.BILINEAR),
                transforms.CombinedRandomHorizontalFlip()]),
        "cityscapesR": transforms.CombinedRandomCrop((768,768)),
    }
    tran_map = {
        "A2D2R": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        "A2D2TNR": transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        "cityscapesmNR": transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        "cityscapesmNR2": transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        "cityscapeslNR": transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        "cityscapeslNR2": transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        "cityscapesNR": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        "cityscapesPR": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        "cityscapesR": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    }
    trans=trans_map.get(dataset_name, None)
    tran=tran_map.get(dataset_name, None)
    return {"transforms": trans,"transform": tran}
