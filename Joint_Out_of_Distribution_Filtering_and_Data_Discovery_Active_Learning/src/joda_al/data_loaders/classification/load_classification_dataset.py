import itertools

from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, SVHN, MNIST
import os
from typing import Dict, List, Tuple

from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter
from joda_al.data_loaders.classification.a2d2_data_loader import get_feature_size, load_A2D2_stream, \
    get_sample_size
from joda_al.data_loaders.classification.classic_dataset import Cacheable_Places365, \
    Cacheable_Places365_Limited, MNISTt2, Places365_ImageListDatset, UnbalancedCIFAR100
from joda_al.data_loaders.classification.tiny_imagenet_dataloader import load_tinyimagenet_pool, \
    TinyImageNet_train
from joda_al.defintions import DataUpdateScenario
from joda_al.data_loaders.classification.ood_detection_datasets import AlteredDataset, NoiseDataset, \
    OpenOODDataset, OODDataset
from joda_al.data_loaders.dataset_registry import A2D2_DATASET_NAME, \
    LARGE_IDENTIFIER, MEDIUM_IDENTIFIER, SMALL_IDENTIFIER, GTAV_DATASET_NAME
from joda_al.data_loaders.classification.gtaV_data_loader import get_feature_size as gta_feature_size, \
    load_gta_v_stream
from joda_al.data_loaders.dataset_lib import update_means, \
    ActiveLearningClassificationDataset, PropertySubset, PropertyConcatDataset
from joda_al.Pytorch_Extentions.dataset_extentions import TransformDataset
from joda_al.utils.logging_utils.log_writers import gl_error


def load_classification_dataset(dataset, config, order=1) -> Tuple[Dataset, List[int], List[int], Dataset, Dataset, Dict, DatasetScenarioConverter]:
    dataset_factory = {
        "A2D2": load_a2d2,
        "TinyImageNet": load_tinyimagenet,
        "GTAVS": load_gtavs,
        "cifar10": load_cifar10,
        "cifar100-ub-ta": load_cifar100_ub_ta,
        "cifar100": load_cifar100,
        "fashionmnist": load_fashionmnist,
        "svhn": load_svhn,
    }
    dataset_comb_name = f'{dataset["name"]}-{dataset["dataset_scenario"]}'
    for key in dataset_factory:
        if dataset_comb_name.startswith(key) or dataset_comb_name in [key, f"{key}-ll", f"{key}-ls", f"{key}-ms", f"{key}-ba", f"{key}-ta", f"{key}-test"]:
            loader = dataset_factory[key]
            break
    else:
        raise NotImplementedError(f"Dataset {dataset_comb_name} not supported.")

    result = loader(dataset, config, order)
    training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, dataset_factory_instance = result

    if (config["data_scenario"] == DataUpdateScenario.osal
        or config["data_scenario"] == DataUpdateScenario.osal_near_far
        or config["data_scenario"] == DataUpdateScenario.osal_extending):
        training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config = apply_open_set_modifications(
            dataset, config, dataset_config, training_pool, label_idx, unlabeled_idcs, validation_set, test_set
        )

    dataset_config["name"] = dataset["name"]
    dataset_config["dataset_scenario"] = dataset["dataset_scenario"]
    if dataset["name"] not in ["cifar10", 'cifar10-test', "cifar10-ll", "cifar100", "cifar100-ll", "cifar100-ls", "cifar100-ms", "svhn"]:
        training_pool = TransformDataset(training_pool, **get_eval_transformations_classification(dataset["name"]))
        validation_set = TransformDataset(validation_set, **get_eval_transformations_classification(dataset["name"]))
        test_set = TransformDataset(test_set, **get_eval_transformations_classification(dataset["name"]))
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, dataset_factory_instance

# --- Helper functions for splitting logic ---

def split_cifar10(dataset_name, data_train):
    if "lls" in dataset_name:
        label_idx = list(range(1000))
        unlabeled_idx = list(range(1000, 10000))
        unlabeled_idcs = [list(range(i * 1000, (i + 1) * 1000)) for i in range(1, 10)]
        val_indices = list(range(45000, 50000))
    elif "test" in dataset_name:
        label_idx = list(range(1000))
        unlabeled_idx = list(range(1000, 2000))
        unlabeled_idcs = [list(range(i * 100, (i + 1) * 100)) for i in range(1, 10)]
        val_indices = list(range(45000, 50000))
    elif "ba" in dataset_name:
        label_idx = list(range(1000))
        unlabeled_idx = list(range(1000, 45000))
        unlabeled_idcs = [list(range(i * 1000, (i + 1) * 1000)) for i in range(1, 44)]
        val_indices = list(range(45000, 50000))
    elif "ll" in dataset_name:
        label_idx = list(range(1000))
        unlabeled_idx = list(range(1000, 45000))
        unlabeled_idcs = [list(range(i * 1000, (i + 1) * 1000)) for i in range(1, 44)]
        val_indices = list(range(45000, 50000))
    elif "lleo" in dataset_name:
        label_idx = list(range(1000))
        unlabeled_idx = list(range(1000, 45000))
        unlabeled_idcs = [list(range(i * 5000 + 1000, min((i + 1) * 5000 + 1000, 45000))) for i in range(0, 9)]
        val_indices = list(range(45000, 50000))
    elif "ms" in dataset_name:
        label_idx = list(range(5000))
        unlabeled_idx = list(range(5000, 45000))
        unlabeled_idcs = [list(range(i * 10000 + 5000, (i + 1) * 10000 + 5000)) for i in range(0, 4)]
        val_indices = list(range(45000, 50000))
    else:
        label_idx = list(range(5000))
        unlabeled_idx = list(range(5000, 45000))
        unlabeled_idcs = [list(range(i * 5000, (i + 1) * 5000)) for i in range(1, 9)]
        val_indices = list(range(45000, 50000))
    return label_idx, unlabeled_idx, unlabeled_idcs, val_indices

def split_cifar100(dataset_name):
    if "ll" in dataset_name:
        label_idx = list(range(1000))
        unlabeled_idx = list(range(1000, 45000))
        unlabeled_idcs = [list(range(i * 1000, (i + 1) * 1000)) for i in range(1, 44)]
        val_indices = list(range(45000, 50000))
    elif "ls" in dataset_name:
        label_idx = list(range(2500))
        unlabeled_idx = list(range(2500, 45000))
        unlabeled_idcs = [unlabeled_idx]
        val_indices = list(range(45000, 50000))
    elif "ta" in dataset_name or "ba" in dataset_name:
        label_idx = list(range(2000))
        unlabeled_idx = list(range(2000, 45000))
        unlabeled_idcs = [list(range(i * 1000, (i + 1) * 1000)) for i in range(2, 44)]
        val_indices = list(range(45000, 50000))
    elif "ms" in dataset_name:
        label_idx = list(range(5000))
        unlabeled_idx = list(range(5000, 45000))
        unlabeled_idcs = [list(range(i * 10000 + 5000, (i + 1) * 10000 + 5000)) for i in range(0, 4)]
        val_indices = list(range(45000, 50000))
    else:
        label_idx = list(range(5000))
        unlabeled_idx = list(range(5000, 45000))
        unlabeled_idcs = [list(range(i * 5000, (i + 1) * 5000)) for i in range(1, 9)]
        val_indices = list(range(45000, 50000))
    return label_idx, unlabeled_idx, unlabeled_idcs, val_indices

# --- Loader functions for each dataset ---

def load_a2d2(dataset, config, order):
    feature_size_no_max, feature_size_resnet, feature_size_vgg = get_feature_size(dataset["name"])
    feature_sizes = set_feature_size(feature_size_no_max, feature_size_resnet, feature_size_vgg)
    dataset_config = {"num_classes": 4, "feature_sizes": feature_sizes, "encoding_dimension": 240, "sample_size": get_sample_size(dataset["name"])}
    training_pool, label_idcx, unlabeled_idcs, validation_set, test_set, dataset_factory = load_A2D2_stream(dataset["path"], order, dataset["name"])
    label_idx = list(itertools.chain(*label_idcx))
    update_means(dataset["name"], label_idx, training_pool, validation_set, test_set)
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, dataset_factory

def load_tinyimagenet(dataset, config, order):
    feature_sizes = set_feature_size([64, 32, 16, 8], [16, 8, 4, 2], [16, 8, 4, 2])
    dataset_config = {"num_classes": 200, "feature_sizes": feature_sizes, "encoding_dimension": 240, "sample_size": (64, 64)}
    if dataset["name"] == "TinyImageNetL-ta":
        feature_sizes = set_feature_size([224, 112, 56, 28], [56, 28, 14, 7], [56, 28, 14, 7])
        dataset_config = {"num_classes": 200, "feature_sizes": feature_sizes, "encoding_dimension": 240, "sample_size": (224, 224)}
    training_pool, label_idcx, unlabeled_idcs, validation_set, test_set, dataset_factory = load_tinyimagenet_pool(dataset["path"], order, dataset["name"])
    label_idx = label_idcx
    update_means(dataset["name"], label_idx, training_pool, validation_set, test_set)
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, dataset_factory

def load_gtavs(dataset, config, order):
    feature_size_no_max, feature_size_resnet, feature_size_vgg = gta_feature_size(dataset["name"])
    feature_sizes = set_feature_size(feature_size_no_max, feature_size_resnet, feature_size_vgg)
    dataset_config = {"num_classes": 4, "feature_sizes": feature_sizes, "encoding_dimension": 240}
    training_pool, label_idcx, unlabeled_idcs, validation_set, test_set, dataset_factory = load_gta_v_stream(dataset["path"], order, dataset["name"])
    label_idx = list(itertools.chain(*label_idcx))
    update_means(dataset["name"], label_idx, training_pool, validation_set, test_set)
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, dataset_factory

def load_cifar10(dataset, config, order):
    test_transform = T.Compose([T.ToTensor()])
    feature_sizes_cifar = set_feature_size([32, 16, 8, 4], [8, 4, 2, 1], [8, 4, 2, 1])
    normalization_params = ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) if "ll" in dataset["name"] or "ms" in dataset["name"] else None
    if normalization_params:
        test_transform.transforms.append(T.Normalize(*normalization_params))
    dataset_config = {"num_classes": 10, "feature_sizes": feature_sizes_cifar, "encoding_dimension": 32, "sample_size": (32, 32)}
    data_train = ActiveLearningClassificationDataset(CIFAR10(os.path.join(dataset["path"], "cifar10"), train=True, download=True, transform=test_transform))
    label_idx, unlabeled_idx, unlabeled_idcs, val_indices = split_cifar10(dataset["name"], data_train)
    training_pool = PropertySubset(data_train, label_idx + unlabeled_idx)
    validation_set = PropertySubset(data_train, val_indices)
    test_set = CIFAR10(os.path.join(dataset["path"], "cifar10"), train=False, download=True, transform=test_transform)
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, DatasetScenarioConverter()

def load_cifar100_ub_ta(dataset, config, order):
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    feature_sizes_cifar = set_feature_size([32, 16, 8, 4], [8, 4, 2, 1], [8, 4, 2, 1])
    dataset_config = {"num_classes": 100, "feature_sizes": feature_sizes_cifar, "encoding_dimension": 32, "sample_size": (32, 32)}
    val_indices = list(range(45000, 50000))
    data_train = ActiveLearningClassificationDataset(
        UnbalancedCIFAR100(os.path.join(dataset["path"], "cifar100"), train=True, download=True,
                           transform=test_transform, ind_classes=config["ind_classes"], near_classes=config["near_classes"], validation_idx=val_indices, ratio=config["near_ratio"]))
    label_idx = data_train.dataset.label_idx
    unlabeled_idx = data_train.dataset.unlabeled_idx
    unlabeled_idcs = data_train.dataset.unlabeled_idcs
    val_indices = data_train.dataset.val_indices
    training_pool = PropertySubset(data_train, label_idx + unlabeled_idx)
    validation_set = PropertySubset(data_train, val_indices)
    test_set = CIFAR100(os.path.join(dataset["path"], "cifar100"), train=False, download=True, transform=test_transform)
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, DatasetScenarioConverter()

def load_cifar100(dataset, config, order):
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    feature_sizes_cifar = set_feature_size([32, 16, 8, 4], [8, 4, 2, 1], [8, 4, 2, 1])
    dataset_config = {"num_classes": 100, "feature_sizes": feature_sizes_cifar, "encoding_dimension": 32, "sample_size": (32, 32)}
    data_train = ActiveLearningClassificationDataset(CIFAR100(os.path.join(dataset["path"], "cifar100"), train=True, download=True, transform=test_transform))
    label_idx, unlabeled_idx, unlabeled_idcs, val_indices = split_cifar100(dataset["name"])
    training_pool = PropertySubset(data_train, label_idx + unlabeled_idx)
    validation_set = PropertySubset(data_train, val_indices)
    test_set = CIFAR100(os.path.join(dataset["path"], "cifar100"), train=False, download=True, transform=test_transform)
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, DatasetScenarioConverter()

def load_fashionmnist(dataset, config, order):
    feature_sizes_cifar = set_feature_size([32, 16, 8, 4], [8, 4, 2, 1], [8, 4, 2, 1])
    dataset_config = {"num_classes": 10, "feature_sizes": feature_sizes_cifar, "encoding_dimension": 28, "sample_size": (28, 28)}
    data_train = FashionMNIST(os.path.join(dataset["path"], "fashionMNIST"), train=True, download=True, transform=T.Compose([T.ToTensor()]))
    test_set = FashionMNIST(os.path.join(dataset["path"], "fashionMNIST"), train=False, download=True, transform=T.Compose([T.ToTensor()]))
    label_idx = list(range(5000))
    unlabeled_idx = list(range(5000, 50000))
    unlabeled_idcs = [list(range(i * 5000, (i + 1) * 5000)) for i in range(1, 9)]
    val_indices = list(range(50000, 60000))
    data_train = ActiveLearningClassificationDataset(data_train)
    training_pool = PropertySubset(data_train, label_idx + unlabeled_idx)
    validation_set = PropertySubset(data_train, val_indices)
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, DatasetScenarioConverter()

def load_svhn(dataset, config, order):
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4496242558551198, 0.4500151436546841, 0.45056651484204796], [0.2007713379168459, 0.19959994740140202, 0.19971907465506])
    ])
    feature_sizes_cifar = set_feature_size([32, 16, 8, 4], [8, 4, 2, 1], [8, 4, 2, 1])
    dataset_config = {"num_classes": 10, "feature_sizes": feature_sizes_cifar, "encoding_dimension": 32, "sample_size": (32, 32)}
    data_train = SVHN(os.path.join(dataset["path"], "svhn"), split='train', download=True, transform=test_transform)
    test_set = SVHN(os.path.join(dataset["path"], "svhn"), split='test', download=True, transform=test_transform)
    label_idx = list(range(5000))
    unlabeled_idx = list(range(5000, 60000))
    unlabeled_idcs = [list(range(i * 5000, (i + 1) * 5000)) for i in range(1, 9)]
    val_indices = list(range(60000, 73257))
    data_train = ActiveLearningClassificationDataset(data_train)
    data_train.targets = data_train.labels
    training_pool = PropertySubset(data_train, label_idx + unlabeled_idx)
    validation_set = PropertySubset(data_train, val_indices)
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config, DatasetScenarioConverter()



def apply_open_set_modifications(dataset : Dict, config : Dict, dataset_config : Dict, data_train, label_idx: List[int], unlabeled_idcs :List[List[int]], validation_set, test_set):
    base_dataset = dataset["name"].split("-")[0]
    ind_classes = config["ind_classes"]
    total_classes = dataset_config["num_classes"]
    dataset_config["num_classes"] = len(ind_classes)

    #filter test and validation datasets
    # val_indices = [idx for idx in val_indices if data_train.targets[idx] in ind_classes]
    # Can be moved to dataset handler aswell
    val_indices = [idx for idx in range(len(validation_set)) if validation_set.targets[idx] in ind_classes]
    test_indices = [idx for idx in range(len(test_set)) if test_set.targets[idx] in ind_classes]

    unlabeled_idx = list(itertools.chain(*unlabeled_idcs))
    all_idx = list(range(len(label_idx) + len(unlabeled_idx)))
    near_classes = list(set(range(total_classes)) - set(ind_classes))
    # adapt to osal scenario to only include ind data in initial labeled set
    if config["data_scenario"] == DataUpdateScenario.osal:
        label_idx = [idx for idx in label_idx if data_train.targets[idx] in ind_classes]


    if config["data_scenario"] == DataUpdateScenario.osal_near_far or config["data_scenario"] == DataUpdateScenario.osal_extending:
        far_dataset = config["far_dataset"]
        near_classes = near_classes if config["near_classes"] is not [] else near_classes
        config["near_classes"] = near_classes
        if far_dataset == "None":
            far_classes = config["far_classes"]
        else:
            len_ind_near = (sum(len(unlabeled) for unlabeled in unlabeled_idcs) + len(label_idx))
            num_streams= len(unlabeled_idcs)

            # Define OOD dataset:
            if far_dataset == "random":
                num_ind_train_samples = int(len(ind_classes) / total_classes * len_ind_near)
                image_size=(3,dataset_config["sample_size"][0],dataset_config["sample_size"][1])
                far_ood = ActiveLearningClassificationDataset(NoiseDataset(num_samples=num_ind_train_samples,image_size=image_size))
            elif far_dataset == "mnist":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(MNIST(os.path.join(dataset["path"], "mnist"), train=False, download=True, transform=get_default_preprocessor(base_dataset)))
            elif far_dataset == "mnistT":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(MNIST(os.path.join(dataset["path"], "mnist"), train=True, download=True, transform=get_default_preprocessor(base_dataset)))
            elif far_dataset == "mnistT2":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(MNISTt2(os.path.join(dataset["path"], "mnist"), train=True, download=True, transform=get_default_preprocessor(base_dataset)))
            elif far_dataset == "places365":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(Cacheable_Places365(os.path.join(dataset["path"], "places365"), split="val", small=True, download=False,
                                                         transform=get_default_preprocessor(base_dataset)))
            elif far_dataset == "places365F":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(Places365_ImageListDatset(os.path.join(dataset["path"], "places365"), ind_dataset=base_dataset, split="val", small=True, download=False,
                                                               transform=get_default_preprocessor(base_dataset)))
            elif far_dataset == "ImageNet800C":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(TinyImageNet_train(os.path.join(dataset["path"], "imagenet-800-C"), transform=get_default_preprocessor(base_dataset)))
            elif far_dataset == "ImageNet800CL":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(TinyImageNet_train(os.path.join(dataset["path"], "imagenet-800-C"), transform=get_default_preprocessor(base_dataset)))
            elif far_dataset == "Textures":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(
                    Cacheable_Places365(os.path.join(dataset["path"], "Textures"), split="val", small=True,
                                        download=False,
                                        transform=get_default_preprocessor(base_dataset)))
            elif far_dataset == "places365T":
                from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = OODDataset(Cacheable_Places365_Limited(os.path.join(dataset["path"], "places365"), split="train-standard", small=True, download=True,
                                                         transform=get_default_preprocessor(base_dataset)))
            else:
                try:
                    from openood.evaluation_api.preprocessor import get_default_preprocessor
                except:
                    gl_error("For OpenSet Active Learning OpenOOD is required for more Datasets.")
                    from joda_al.data_loaders.classification.ood_detection_datasets import get_default_preprocessor
                far_ood = ActiveLearningClassificationDataset(OpenOODDataset(
                    root=os.path.join(dataset["path"], "ood_data"),
                    name='_'.join((f'{far_dataset}', 'ood', 'far')),
                    imglist_pth=os.path.join(dataset["path"], "ood_data", f'benchmark_imglist/{base_dataset}/test_{far_dataset}.txt'),
                    data_dir=os.path.join(dataset["path"], "ood_data", 'images_classic/'),
                    num_classes=10,
                    preprocessor=get_default_preprocessor(base_dataset),
                    data_aux_preprocessor=get_default_preprocessor(base_dataset)
                ))


            far_classes = [-1]
            config["far_classes"].extend(far_classes)
            # Calculate some stupid ratios:
            ind_to_farood = (len_ind_near * len(ind_classes) / total_classes) / len(far_ood)
            len_ind_pool=len(data_train)  #Assume OOD data is appended after the InD Data
            labeled_farood = list(range(len_ind_pool, len_ind_pool + int(len(ind_classes)/total_classes * len(label_idx)/ind_to_farood)))

            if config["data_scenario"] == DataUpdateScenario.osal_extending:
                dataset_config["all_classes"] = total_classes
                label_idx = [idx for idx in label_idx if data_train.targets[idx] in ind_classes]

            ind_ood_label_idx = label_idx + labeled_farood
            normal_size = (len(far_ood) - len(labeled_farood)) // num_streams
            far_ood_idcs = [list(range(i * normal_size + len_ind_pool+ len(labeled_farood), # starting from the end of the labeled farood
                            min((i + 1) * normal_size + len_ind_pool + len(labeled_farood), # up to the end of the farood
                            len(data_train) + len(far_ood))))
                            for i in range(num_streams)]

            ind_ood_unlabeled_idcs = [x + y for x, y in zip(unlabeled_idcs, far_ood_idcs)]
            ind_ood_data_train = PropertyConcatDataset([data_train, far_ood])
            # unlabeled_farood = list(set(labeled_farood).symmetric_difference(set(range(len(data_train), len(data_train) + len(far_ood)))))
            # unlabeled_idx = unlabeled_idx + unlabeled_farood
            # all_idx = all_idx + list(range(len(all_idx), len(all_idx) + len(far_ood)))

            # Update Existing
            # all_idx = all_idx + labeled_farood
            label_idx = ind_ood_label_idx
            unlabeled_idcs = ind_ood_unlabeled_idcs
            data_train=ind_ood_data_train

    training_pool = AlteredDataset(data_train,
                                ind_classes=ind_classes, near_classes=near_classes, far_classes=far_classes if (config["data_scenario"] == DataUpdateScenario.osal_near_far or config["data_scenario"] == DataUpdateScenario.osal_extending) else [])
    validation_set = AlteredDataset(PropertySubset(validation_set, val_indices),
                                    ind_classes=ind_classes, near_classes=near_classes, far_classes=far_classes if (config["data_scenario"] == DataUpdateScenario.osal_near_far or config["data_scenario"] == DataUpdateScenario.osal_extending) else [])
    test_set = AlteredDataset(PropertySubset(test_set, test_indices),
                              ind_classes=ind_classes, near_classes=near_classes, far_classes=far_classes if (config["data_scenario"] == DataUpdateScenario.osal_near_far or config["data_scenario"] == DataUpdateScenario.osal_extending) else [])
    return training_pool, label_idx, unlabeled_idcs, validation_set, test_set, dataset_config


def update_altered_dataset_map(altered_dataset, ind_classes, near_classes, far_classes):
    if isinstance(altered_dataset, TransformDataset):
        if isinstance(altered_dataset.dataset, AlteredDataset):
            altered_dataset.dataset.set_map(ind_classes, near_classes, far_classes)
        else:
            raise NotImplementedError("Altered Dataset Required")
    elif isinstance(altered_dataset, AlteredDataset):
        altered_dataset.set_map(ind_classes, near_classes, far_classes)
    else:
        raise NotImplementedError("Altered Dataset Required")


def reinit_altered_propertysubset_set(dataset,config,dataset_config):
    ind_classes = config.experiment_config["ind_classes"]
    near_classes = config.experiment_config["near_classes"]
    far_classes = config.experiment_config["far_classes"]
    # check AlteredDataset
    #Can either a
    # TransformDataset -> AlteredDataset -> PropertySubSet -> Dataset
    if isinstance(dataset, TransformDataset):
        if isinstance(dataset.dataset,AlteredDataset):
            if isinstance(dataset.dataset.dataset, PropertySubset):
                dataset_indices = [idx for idx in range(len(dataset.dataset.dataset.dataset)) if dataset.dataset.dataset.dataset.targets[idx] in ind_classes]
                newset=TransformDataset(AlteredDataset(PropertySubset(dataset.dataset.dataset.dataset, dataset_indices),
                               ind_classes=ind_classes, near_classes=near_classes, far_classes=far_classes if (
                            config.experiment_config["data_scenario"] == DataUpdateScenario.osal_near_far or config.experiment_config[
                        "data_scenario"] == DataUpdateScenario.osal_extending) else []), **get_eval_transformations_classification(dataset_config["name"]))

    # TransformDataset -> AlteredDataset -> Dataset
            else:
                newset=TransformDataset(AlteredDataset(dataset,
                                                ind_classes=ind_classes, near_classes=near_classes,
                                                far_classes=far_classes if (
                                                        config.experiment_config["data_scenario"] == DataUpdateScenario.osal_near_far or
                                                        config.experiment_config[
                                                            "data_scenario"] == DataUpdateScenario.osal_extending) else []),
                                 **get_eval_transformations_classification(dataset["name"]))
        else:
            raise NotImplementedError ("Unsupported")
    # AlteredDataset -> PropertySubSet -> Dataset
    elif isinstance(dataset,AlteredDataset):
        if isinstance(dataset.dataset, PropertySubset):
            dataset_indices = [idx for idx in range(len(dataset.dataset.dataset)) if dataset.dataset.dataset.targets[idx] in ind_classes]
            newset = AlteredDataset(PropertySubset(dataset.dataset.dataset, dataset_indices),
                                    ind_classes=ind_classes, near_classes=near_classes, far_classes=far_classes if (
                        config.experiment_config["data_scenario"] == DataUpdateScenario.osal_near_far or config.experiment_config[
                    "data_scenario"] == DataUpdateScenario.osal_extending) else [])
        # AlteredDataset -> Dataset
        else:
            newset = AlteredDataset(dataset.dataset,
                                    ind_classes=ind_classes, near_classes=near_classes, far_classes=far_classes if (
                        config.experiment_config["data_scenario"] == DataUpdateScenario.osal_near_far or config.experiment_config[
                    "data_scenario"] == DataUpdateScenario.osal_extending) else [])
    # PropertySubSet
    elif isinstance(dataset,PropertySubset):
        dataset_indices = [idx for idx in range(len(dataset.dataset)) if dataset.dataset.targets[idx] in ind_classes]
        newset = AlteredDataset(PropertySubset(dataset.dataset, dataset_indices),
                                ind_classes=ind_classes, near_classes=near_classes, far_classes=far_classes if (
                    config.experiment_config["data_scenario"] == DataUpdateScenario.osal_near_far or config.experiment_config[
                "data_scenario"] == DataUpdateScenario.osal_extending) else [])
    else:
        raise NotImplementedError("Validation/Testset in OpenSetActive Learning must be indexable TransformDataset -> AlteredDataset -> PropertySubSet -> Dataset or AlteredDataset -> PropertySubSet -> Dataset or AlteredDataset -> Dataset")

    return newset


def get_training_transformations_classification(dataset_name):
    trans_map = {
        "A2D2": None,
        "A2D2M": None,
        "A2D2N": None,
        "A2D2E": None,
        "A2D2mNR": T.Compose([T.RandomHorizontalFlip(), T.RandomAdjustSharpness(2), T.GaussianBlur(3)]),
        "A2D2NR": T.RandomResizedCrop((151, 240), (0.7, 1), ratio=(240 / 152, 240 / 150)),
        "A2D2NR3": T.RandomResizedCrop((151, 240), (0.75, 1), ratio=(4 / 3, 16 / 9)),
        "A2D2R": T.RandomResizedCrop((151, 240), (0.6, 1), ratio=(4 / 3, 16 / 9)),
        "A2D2ER": T.RandomResizedCrop((151, 240), (0.6, 1), ratio=(4 / 3, 16 / 9)),
        "A2D2R2": T.Compose(
            [T.RandomResizedCrop((151, 240), (0.7, 1), ratio=(240 / 152, 240 / 150)), T.RandomHorizontalFlip()]),
        "A2D2NR2": T.Compose(
            [T.RandomResizedCrop((151, 240), (0.75, 1), ratio=(240 / 152, 240 / 150)), T.RandomHorizontalFlip(),
             T.RandomAdjustSharpness(2), T.GaussianBlur(3)]),
        "A2D2ER2": T.Compose(
            [T.RandomResizedCrop((151, 240), (0.7, 1), ratio=(240 / 152, 240 / 150)), T.RandomHorizontalFlip(),
             T.RandomAdjustSharpness(2), T.GaussianBlur(3)]),
        "nuscenesNR": T.Compose([T.RandomHorizontalFlip(), T.RandomAdjustSharpness(2), T.GaussianBlur(3)]),
        "cityscapesNR": T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(0.4, 0.4, 0.4)]),
        "cityscapesER": T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(0.4, 0.4, 0.4)]),
        "cifar10-ll": T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4)]),
        "cifar100-ms": T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4)]),
        "cifar100-ta": T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4)]),
        "cifar100-ub-ta": T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4)]),
        "TinyImageNet-ta": T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=64, padding=4)]),
        "TinyImageNetL-ta": T.Compose(
            [T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomCrop(size=224, padding=4)]),
        "TinyImageNet-be": T.Compose([T.RandomHorizontalFlip(), T.RandomResizedCrop(size=64)]),
    }
    trans=trans_map.get(dataset_name, None)
    return {"transform":trans,"transforms": None}


def get_eval_transformations_classification(dataset_name):
    trans_map = {
        "A2D2l": T.Resize((144, 240)),
        "A2D2m": T.Resize((72, 120)),
        "A2D2s": T.Resize((36, 60)),
        "A2D2lE": T.Resize((144, 240)),
        "A2D2mE": T.Resize((72, 120)),
        "A2D2sE": T.Resize((36, 60)),
        "A2D2lU": T.Resize((144, 240)),
        "A2D2mU": T.Resize((72, 120)),
        "A2D2sU": T.Resize((36, 60)),
        "A2D2lN": T.Resize((144, 240)),
        "A2D2mN": T.Resize((72, 120)),
        "A2D2sN": T.Resize((36, 60)),
        "A2D2mNR": T.Resize((72, 120)),
        "GTAVSm": T.Resize((72, 128)),
        "GTAVSmN": T.Resize((72, 128)),
        "TinyImageNetL-ta": T.Resize((224, 224)),
    }
    transformation=trans_map.get(dataset_name, None)
    if transformation is None:
        if A2D2_DATASET_NAME in dataset_name:
            if LARGE_IDENTIFIER in dataset_name:
                transformation=T.Resize((144, 240))
            elif MEDIUM_IDENTIFIER in dataset_name:
                transformation = T.Resize((72, 120))
            elif SMALL_IDENTIFIER in dataset_name:
                transformation = T.Resize((36, 60))
        elif GTAV_DATASET_NAME in dataset_name:
            if MEDIUM_IDENTIFIER in dataset_name:
                transformation=T.Resize((72, 128))
    return {"transform": transformation}


def set_feature_size(feature_sizes_no_max, feature_size_resnet,feature_sizes_vgg):
    feature_sizes = {
        "Resnet18": feature_sizes_no_max,
        "Resnet18C": feature_sizes_no_max,
        "Resnet18D": feature_sizes_no_max,
        "Resnet34": feature_sizes_no_max,
        "Resnet18M": feature_sizes_no_max,
        "Resnet34M": feature_sizes_no_max,
        "Resnet18FM": feature_sizes_no_max,
        "Resnet34FM": feature_sizes_no_max,
        "Resnet18T": feature_size_resnet,
        "Resnet34T": feature_size_resnet,
        "Resnet18F": feature_size_resnet,
        "Resnet34F": feature_size_resnet,
        "Resnet50": feature_sizes_no_max,
        "Resnet50T": feature_sizes_no_max,
        "VGG11": feature_sizes_vgg,
        "VGG11T": feature_sizes_vgg,
        "VGG11F": feature_sizes_vgg,
        "VGG16F": feature_sizes_vgg,
        "VGG11bnT": feature_sizes_vgg,
        "VGG11bnR": feature_sizes_vgg,
        "VGG11bnO": feature_sizes_vgg,
        "VGG11bnF": feature_sizes_vgg,
        "VGG16bnF": feature_sizes_vgg,
        "VGG16bnT": feature_sizes_vgg,
        "VGG16": feature_sizes_vgg
    }
    return feature_sizes

