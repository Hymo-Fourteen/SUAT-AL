import os
from typing import Dict, Callable, TypeVar, Tuple, Optional, Any

import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from joda_al.Pytorch_Extentions.scheduler import get_scheduler
from joda_al.models.classification.model_utils import setup_classification_model
from joda_al.models.model_lib import CompoundModel, EnsembleModel
from joda_al.task_supports.task_handler_factory import get_task_handler
from joda_al.defintions import TaskDef, ActiveLearningTrainingStrategies
from joda_al.models.SemanticSegmentation.model_utils import setup_seg_model, get_optimizer_scheduler_semseg
from joda_al.models.loss_functions import LOSS_FUNCTIONS
from joda_al.utils.logging_utils.log_writers import gl_info
from joda_al.utils.method_utils import is_loss_learning, is_method_of

LossFunc = TypeVar("LossFunc", bound=Callable)


def save_model(
    epoch: int,
    model: Dict[str, nn.Module],
    optimizer: Dict[str, Optimizer],
    scheduler: Dict[str, Optional[_LRScheduler]],
    loss: Any,
    path: str,
    model_name: str = "checkpoint.model"
) -> None:
    """
    Save the current model, optimizer, scheduler, and loss state to a given path.

    Args:
        epoch (int): Current epoch.
        model (Dict[str, nn.Module]): Dictionary of model modules.
        optimizer (Dict[str, Optimizer]): Dictionary of optimizers.
        scheduler (Dict[str, Optional[_LRScheduler]]): Dictionary of schedulers.
        loss (Any): Loss value or object to save.
        path (str): Directory to save the checkpoint.
        model_name (str): Name of the checkpoint file.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': {m_n: m.state_dict() for m_n, m in model.items()},
        'optimizer_state_dict': {opt_n: opt.state_dict() for opt_n, opt in optimizer.items()},
        "scheduler_state_dict": {sdl_n: sdl.state_dict() for sdl_n, sdl in scheduler.items() if sdl is not None},
        'loss': loss
    }, os.path.join(path, model_name))


def slim_save_model(
    model: Dict[str, nn.Module],
    path: str,
    model_name: str = "pure_checkpoint.model"
) -> None:
    """
    Save only the model state dicts to a given path.

    Args:
        model (Dict[str, nn.Module]): Dictionary of model modules.
        path (str): Directory to save the checkpoint.
        model_name (str): Name of the checkpoint file.
    """
    torch.save({
        'model_state_dict': {m_n: m.state_dict() for m_n, m in model.items()},
    }, os.path.join(path, model_name))


def slim_load_model(
    models: CompoundModel,
    path: str,
    model_name: str = "pure_checkpoint.model"
) -> CompoundModel:
    """
    Load model weights from a slim checkpoint.

    Args:
        models (CompoundModel): Model(s) to load weights into.
        path (str): Directory containing the checkpoint.
        model_name (str): Name of the checkpoint file.

    Returns:
        CompoundModel: Models with loaded weights.
    """
    models, _, _, _, _ = load_model_with_model(models, {}, {}, path, model_name=model_name)
    return models


def load_model(
    dataset_config: Dict,
    training_config: Dict,
    task: str,
    device: Any,
    method: str,
    path: str,
    cls_weights: Optional[torch.Tensor] = None,
    cls_counts: Optional[Any] = None,
    model_name: str = "checkpoint.model"
) -> Tuple[_Loss, Dict[str, nn.Module], Dict[str, Optimizer], Dict[str, Optional[_LRScheduler]], int, Any]:
    """
    Load a model and its optimizer, scheduler, and loss state from a checkpoint.

    Args:
        dataset_config (Dict): Dataset configuration.
        training_config (Dict): Training configuration.
        task (str): Task type.
        device (Any): Device to load the model on.
        method (str): Method name.
        path (str): Directory containing the checkpoint.
        cls_weights (Optional[torch.Tensor]): Class weights.
        cls_counts (Optional[Any]): Class counts.
        model_name (str): Name of the checkpoint file.

    Returns:
        Tuple containing criterion, models, optimizers, schedulers, epoch, and loss.
    """
    criterion, models, optimizers, schedulers = setup_model_optimizer_loss(
        dataset_config, training_config, task, device, method, cls_weights, cls_counts
    )
    models, optimizers, schedulers, epoch, loss = load_model_with_model(
        models, optimizers, schedulers, path, model_name=model_name
    )
    return criterion, models, optimizers, schedulers, epoch, loss


def load_model_with_model(
    models: Dict[str, nn.Module],
    optimizers: Dict[str, Optimizer],
    schedulers: Dict[str, Optional[_LRScheduler]],
    path: str,
    model_name: str = "checkpoint.model"
) -> Tuple[Dict[str, nn.Module], Dict[str, Optimizer], Dict[str, Optional[_LRScheduler]], int, Any]:
    """
    Load model, optimizer, and scheduler state dicts from a checkpoint.

    Args:
        models (Dict[str, nn.Module]): Model(s) to load weights into.
        optimizers (Dict[str, Optimizer]): Optimizer(s) to load state into.
        schedulers (Dict[str, Optional[_LRScheduler]]): Scheduler(s) to load state into.
        path (str): Directory containing the checkpoint.
        model_name (str): Name of the checkpoint file.

    Returns:
        Tuple of models, optimizers, schedulers, epoch, and loss.
    """
    checkpoint = torch.load(os.path.join(path, model_name))
    for m_name, m in models.items():
        m.load_state_dict(checkpoint['model_state_dict'][m_name])
    if "optimizer_state_dict" in checkpoint:
        for opt_name, opt in optimizers.items():
            opt.load_state_dict(checkpoint['optimizer_state_dict'][opt_name])
    if "scheduler_state_dict" in checkpoint:
        for sdl_name, sdl in schedulers.items():
            if sdl is not None:
                sdl.load_state_dict(checkpoint['scheduler_state_dict'][sdl_name])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0)
    return models, optimizers, schedulers, epoch, loss


def setup_loss_function(
    method: str,
    training_config: Dict,
    dataset_config: Dict,
    task: str,
    cls_weights: Optional[torch.Tensor],
    device: Any
) -> _Loss:
    """
    Setup the loss function for the given task and configuration.

    Args:
        method (str): Method name.
        training_config (Dict): Training configuration.
        dataset_config (Dict): Dataset configuration.
        task (str): Task type.
        cls_weights (Optional[torch.Tensor]): Class weights.
        device (Any): Device to use.

    Returns:
        _Loss: Loss function instance.
    """
    weights = cls_weights.to(device) if training_config.get("cls_weights") == "all" and cls_weights is not None else None
    loss_func_name = training_config["loss_func"]
    if loss_func_name in ["CrossEntropy", "GorLoss"]:
        criterion = LOSS_FUNCTIONS[task].get(loss_func_name, None)(weight=weights, reduction='none')
    else:
        criterion = LOSS_FUNCTIONS[task].get(loss_func_name, None)(weight=weights, reduction='none', **training_config)
    if criterion is None:
        raise NotImplementedError(f'Loss function {loss_func_name} not implemented')
    return criterion


def setup_model_optimizer_loss(
    dataset_config: Dict,
    training_config: Dict,
    task: str,
    device: Any,
    method: str,
    cls_weights: Optional[torch.Tensor] = None,
    cls_counts: Optional[Any] = None
) -> Tuple[_Loss, Dict[str, nn.Module], Dict[str, Optimizer], Dict[str, Optional[_LRScheduler]]]:
    """
    Setup models, optimizers, schedulers, and loss function for a given task and configuration.

    Args:
        dataset_config (Dict): Dataset configuration.
        training_config (Dict): Training configuration.
        task (str): Task type.
        device (Any): Device to use.
        method (str): Method name.
        cls_weights (Optional[torch.Tensor]): Class weights.
        cls_counts (Optional[Any]): Class counts.

    Returns:
        Tuple of criterion, models, optimizers, and schedulers.
    """
    models = setup_task_models(task, method, device, dataset_config, training_config, cls_counts)
    criterion = get_task_handler(task).setup_loss_function(method, training_config, dataset_config, task, cls_weights, device)
    optimizers: Dict[str, Optimizer] = {}
    schedulers: Dict[str, Optional[_LRScheduler]] = {}

    if training_config.get("ensemble", False):
        for i, model in enumerate(models["task"]):
            optim_backbone, sched_backbone = get_optimizer_scheduler_by_task(model, training_config, task)
            optimizers[f"task_{i}"] = optim_backbone
            schedulers[f"task_{i}"] = sched_backbone
    else:
        optim_backbone, sched_backbone = get_optimizer_scheduler_by_task(models["task"], training_config, task)
        optimizers["task"] = optim_backbone
        schedulers["task"] = sched_backbone

    if is_loss_learning(method):
        ll_lr = training_config["loss_module_lr"] if training_config["loss_module_lr"] > 0 else training_config["lr"]
        optim_module, sched_module = get_optimizer_scheduler_by_task(models['module'], training_config, task, ll_lr)
        optimizers["module"] = optim_module
        schedulers["module"] = sched_module
    elif is_method_of(method, ["LfOSA"]):
        optim_module, sched_module = get_optimizer_scheduler_by_task(models['module'], training_config, task)
        optimizers["module"] = optim_module
        schedulers["module"] = sched_module

    return criterion, models, optimizers, schedulers


def get_optimizer_scheduler_by_task(
    model: nn.Module,
    training_config: Dict,
    task: str,
    lr: Optional[float] = None
) -> Tuple[Optimizer, Optional[_LRScheduler]]:
    """
    Get optimizer and scheduler for a given task.

    Args:
        model (nn.Module): Model to optimize.
        training_config (Dict): Training configuration.
        task (str): Task type.
        lr (Optional[float]): Learning rate.

    Returns:
        Tuple of optimizer and scheduler.
    """
    if task == TaskDef.classification or task == TaskDef.zeroclassification:
        return get_optimizer_scheduler_classification(model, training_config, lr)
    elif task == TaskDef.semanticSegmentation:
        return get_optimizer_scheduler_semseg(model, training_config)
    else:
        raise NotImplementedError(f"Optimizer/scheduler for task {task} not implemented.")


def get_optimizer_scheduler_classification(
    model: nn.Module,
    training_config: Dict,
    lr: Optional[float] = None
) -> Tuple[Optimizer, Optional[_LRScheduler]]:
    """
    Get optimizer and scheduler for classification tasks.

    Args:
        model (nn.Module): Model to optimize.
        training_config (Dict): Training configuration.
        lr (Optional[float]): Learning rate.

    Returns:
        Tuple of optimizer and scheduler.
    """
    lr = lr if lr is not None else training_config["lr"]
    optimizer: Optimizer
    scheduler: Optional[_LRScheduler] = None

    if training_config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=training_config["momentum"],
            weight_decay=training_config["wdcay"]
        )
        scheduler = get_scheduler(training_config, optimizer)
    elif training_config["optimizer"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=training_config["wdcay"]
        )
    elif training_config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=training_config["wdcay"],
            momentum=training_config["momentum"]
        )
    else:
        raise NotImplementedError(f"Optimizer {training_config['optimizer']} not implemented.")
    return optimizer, scheduler


def setup_task_models(
    task: str,
    method: str,
    device: Any,
    dataset_config: Dict,
    training_config: Dict,
    cls_counts: Optional[Any] = None
) -> Dict[str, nn.Module]:
    """
    Setup models for a given task.

    Args:
        task (str): Task type.
        method (str): Method name.
        device (Any): Device to use.
        dataset_config (Dict): Dataset configuration.
        training_config (Dict): Training configuration.
        cls_counts (Optional[Any]): Class counts.

    Returns:
        Dict[str, nn.Module]: Dictionary of models.
    """
    if task == TaskDef.classification:
        return setup_classification_model(method, device, dataset_config, training_config, cls_counts)
    elif task == TaskDef.semanticSegmentation:
        return setup_seg_model(method, device, dataset_config, training_config, cls_counts)
    else:
        raise NotImplementedError(f"Task {task} not implemented.")


def setup_model_optimizer_by_strategy(
    experiment_config: Dict,
    dataset_config: Dict,
    training_config: Dict,
    task: str,
    device: Any,
    method: str,
    cls_weights: Optional[torch.Tensor],
    cls_counts: Optional[Any],
    model_name: Optional[str] = None,
) -> Tuple[_Loss, Dict[str, nn.Module], Dict[str, Optimizer], Dict[str, Optional[_LRScheduler]]]:
    """
    Setup model, optimizer, scheduler, and loss function according to the active learning training strategy.

    Args:
        experiment_config (Dict): Experiment configuration.
        dataset_config (Dict): Dataset configuration.
        training_config (Dict): Training configuration.
        task (str): Task type.
        device (Any): Device to use.
        method (str): Method name.
        cls_weights (Optional[torch.Tensor]): Class weights.
        cls_counts (Optional[Any]): Class counts.
        model_name (Optional[str]): Name of the checkpoint file.

    Returns:
        Tuple of criterion, models, optimizers, and schedulers.
    """
    if model_name is None:
        model_name = "checkpoint.model"
    criterion, models, optimizers, schedulers = setup_model_optimizer_loss(
        dataset_config, training_config, task, device, method, cls_weights, cls_counts
    )
    experiment_path = experiment_config["experiment_path"]
    strategy = experiment_config["strategy"]

    if strategy == ActiveLearningTrainingStrategies.retrain:
        gl_info("Creating New Model")
        return criterion, models, optimizers, schedulers
    elif strategy == ActiveLearningTrainingStrategies.reuse:
        if os.path.exists(os.path.join(experiment_path, "checkpoint.model")):
            gl_info("Model to reuse found loading weights")
            models, _, _, _, _ = load_model_with_model(
                models, {}, {}, experiment_path, model_name=model_name
            )
        else:
            gl_info("No Model to reuse start from scratch")
    elif strategy == ActiveLearningTrainingStrategies.continuous:
        if os.path.exists(os.path.join(experiment_path, "checkpoint.model")):
            gl_info("Model to reuse found loading weights and optimizer")
            models, optimizers, schedulers, epoch, loss = load_model_with_model(
                models, optimizers, schedulers, experiment_path, model_name=model_name
            )
        else:
            gl_info("No Model to continue start from scratch")
    else:
        raise NotImplementedError("Current Training Strategy is not implemented")
    return criterion, models, optimizers, schedulers
