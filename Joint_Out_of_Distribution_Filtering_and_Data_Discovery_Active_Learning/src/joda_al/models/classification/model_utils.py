import os
import torch
from typing import Dict, Any, Optional

from joda_al.models import custom_models as resnet, torchResnet
from joda_al.models.model_lib import EnsembleModel
from joda_al.models.query_models.query_models import GenericLossNet
from joda_al.utils.method_utils import is_loss_learning, is_method_of
import copy


def setup_classification_model(
    method: str,
    device: Any,
    dataset_config: Dict,
    training_config: Dict,
    cls_counts: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Set up classification model(s) and any required modules for a given method and configuration.

    Args:
        method (str): The active learning method.
        device (Any): The CUDA device.
        dataset_config (Dict): Dataset configuration.
        training_config (Dict): Training configuration.
        cls_counts (Optional[Any]): Optional class counts.

    Returns:
        Dict[str, Any]: Dictionary containing the task model and any additional modules.
    """
    with torch.cuda.device(device):
        if training_config.get("ensemble", False) is True:
            task_model = classification_ensemble_factory(dataset_config, method, training_config)
        else:
            task_model = classification_model_factory(dataset_config, method, training_config)

        models = {"task": task_model}
        if is_loss_learning(method):
            feature_sizes = dataset_config["feature_sizes"][training_config["model"]]
            loss_module = GenericLossNet(feature_sizes, task_model.num_channels).cuda()
            models['module'] = loss_module
        elif is_method_of(method, ["LfOSA"]):
            detector_dataset_config = copy.deepcopy(dataset_config)
            detector_dataset_config["num_classes"] = dataset_config["num_classes"] + 1
            detector = classification_model_factory(detector_dataset_config, method, training_config)
            models["module"] = detector
        if training_config["load_from_checkpoint"] != "":
            models = load_pretrained_checkpoint(models, training_config)
        return models


def load_pretrained_checkpoint(
    models: Dict[str, Any],
    config: Dict
) -> Dict[str, Any]:
    """
    Load pretrained weights into the task model from a checkpoint.

    Args:
        models (Dict[str, Any]): Dictionary containing the task model.
        config (Dict): Training configuration containing checkpoint path.

    Returns:
        Dict[str, Any]: Updated models with loaded weights.
    """
    model_name = config["load_from_checkpoint"]
    checkpoint_loaded = torch.load(os.path.join(model_name))
    checkpoint = {k.split("backbone.")[-1]: v for k, v in checkpoint_loaded.items()}
    models["task"].load_state_dict(checkpoint, strict=False)
    return models


def classification_ensemble_factory(
    dataset_config: Dict,
    method: str,
    training_config: Dict,
) -> EnsembleModel:
    """
    Create an ensemble of classification models.

    Args:
        dataset_config (Dict): Dataset configuration.
        method (str): The active learning method.
        training_config (Dict): Training configuration.

    Returns:
        EnsembleModel: An ensemble of models.
    """
    models = []
    dataset_config_copy = copy.deepcopy(dataset_config)
    if method in ["aol"]:
        dataset_config_copy["num_classes"] = dataset_config["num_classes"] + 1
    for _ in range(training_config["ensemble_size"]):
        task_model = classification_model_factory(dataset_config_copy, method, training_config)
        models.append(task_model)
    return EnsembleModel(models)


def classification_model_factory(
    dataset_config: Dict,
    method: str,
    training_config: Dict,
) -> Any:
    """
    Factory for creating a classification model based on configuration.

    Args:
        dataset_config (Dict): Dataset configuration.
        method (str): The active learning method.
        training_config (Dict): Training configuration.
        cls_head (Optional[Any]): Optional classification head.

    Returns:
        Any: Instantiated model.
    """

    #Make sure that the moodel has
    if training_config["loss_func"] == "GorLoss":
        norm_layer_type = "gn"
    else:
        norm_layer_type = "bn"

    if training_config["model"] == "Resnet18":
        task_model = resnet.ResNet18(num_classes=dataset_config["num_classes"],norm_layer_type=norm_layer_type,**training_config["model_configuration"])
    elif training_config["model"] == "Resnet18Cls":
        task_model = resnet.ResNet18Cls(num_classes=dataset_config["num_classes"],norm_layer_type=norm_layer_type,**training_config["model_configuration"])
    elif training_config["model"] == "Resnet34":
        task_model = resnet.ResNet34(num_classes=dataset_config["num_classes"],norm_layer_type=norm_layer_type,**training_config["model_configuration"])
    elif training_config["model"] == "Resnet50":
        task_model = resnet.ResNet50(num_classes=dataset_config["num_classes"],norm_layer_type=norm_layer_type,**training_config["model_configuration"])
    else:
        raise NotImplementedError(f"model {training_config['model']} not implemented")
    return task_model.cuda()
