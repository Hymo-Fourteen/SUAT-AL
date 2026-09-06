from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union

from joda_al.Config.config_def import NestedConfig
from joda_al.Config.method_configs.joda_config import AbstractSelectionMethodConfig, SelectionMethodConfig
from joda_al.Config.method_configs.query_config import QueryMethodConfig


@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Attributes:
        lr (float): Learning rate for the optimizer. Default is 1e-1.
        lr_backbone (float): Learning rate for the backbone network. Default is 1e-3.
        momentum (float): Momentum factor for optimizers like SGD. Default is 0.9.
        wdcay (float): Weight decay (L2 penalty). Default is 5e-4.
        batch_size (int): Number of samples per batch. Default is 128.
        num_epochs (int): Number of training epochs. Default is 200.
        softmax (bool): Whether to apply softmax to outputs. Default is False.
        optimizer (str): Optimizer type (e\.g\., "SGD", "Adam"). Default is "SGD".
        early_stopping (str): Metric for early stopping. Default is "val_acc".
        patience (int): Number of epochs to wait for improvement before stopping. Default is 200.
        num_workers (int): Number of data loading workers. Default is 4.
        cls_weights (Optional[str]): Path to class weights file or None. Default is None.
        model (str): Model architecture to use. Default is "Resnet18".
        pretrained (bool): Use pretrained weights for the model. Default is False.
        pretrained_backbone (bool): Use pretrained weights for the backbone. Default is True.
        aux_loss (bool): Use auxiliary loss if available. Default is False.
        milestones (List[int]): Epochs at which to change learning rate. Default is empty list.
        scheduler (Dict): Scheduler configuration. Default is {"type": "cosine"}.
        warm_up (bool): Use learning rate warm-up. Default is False.
        epoch_start (int): Starting epoch number. Default is 0.
        epsilon (float): Epsilon value for loss or optimizer. Default is 0.1.
        eval_frequencey (int): Frequency of evaluation (in epochs). Default is 1.
        loss_func (str): Loss function to use. Default is "CrossEntropy".
        alpha (float): Alpha parameter for loss or regularization. Default is 0.5.
        triplet_margin (float): Margin for triplet loss. Default is 0.5.
        triplet_mode (str): Triplet loss mode ("off" to disable). Default is "off".
        store_predictions (bool): Whether to store model predictions. Default is False.
        ensemble (bool): Use model ensembling. Default is False.
        ensemble_size (int): Number of models in the ensemble. Default is 3.
        ensemble_training (bool): Enable ensemble training. Default is True.
        load_from_checkpoint (str): Path to checkpoint for loading weights. Default is "".
    """
    lr: float = 1e-1
    lr_backbone: float = 1e-3
    momentum: float = 0.9
    wdcay: float = 5e-4
    batch_size: int = 128
    num_epochs: int = 200
    softmax: bool = False
    optimizer: str = "SGD"
    early_stopping: str = "val_acc"
    patience: int = 200
    num_workers: int = 4
    cls_weights: Optional[str] = None
    aux_loss: bool = False
    milestones: List[int] = field(default_factory=list)
    scheduler: Dict = field(default_factory=lambda: {"type": "cosine"})
    warm_up: bool = False
    epoch_start: int = 0
    epsilon: float = 0.1 #Doppelt Belegt
    eval_frequencey: int = 1
    loss_func: str = "CrossEntropy"
    alpha: float = 0.5 #Doppelt Belegt
    triplet_margin: float = 0.5
    triplet_mode: str = "off"
    store_predictions: bool = False
    ensemble: bool = False
    ensemble_size: int = 3
    ensemble_training: bool = True #True
    load_from_checkpoint: str = ""

@dataclass
class ModelSubConfig:
    do_ratio: float = 0.05
    do_training: bool = False

@dataclass
class ModelConfig:
    """
    Configuration for model architecture.

    Attributes:
        model (str): Model architecture to use. Default is "Resnet18".
        pretrained (bool): Use pretrained weights for the model. Default is False.
        pretrained_backbone (bool): Use pretrained weights for the backbone. Default is True.
        do_ratio (float): Dropout ratio or similar. Default is 0.05.
        do_training (bool): If Dropout is applied during training.
        aux_loss (bool): Use auxiliary loss if available. Default is False.
    """
    model: str = "Resnet18"
    pretrained: bool = False
    pretrained_backbone: bool = True
    model_configuration: Union[Dict,ModelSubConfig] = field(default_factory=lambda: {})



@dataclass
class ExperimentConfig:
    """
    Configuration for experiment setup.

    Parameters:
        method_type (str): The active learning method type to use. Default is "lloss".
        skip_first (bool): Whether to skip the first cycle. Default is False.
        resume (Optional[int]): Resume from a specific checkpoint or cycle. Default is None.
        checkpoint_path (str): Path to the checkpoint file. Default is "/src/checkpoint.model".
        only_select (bool): If True, only perform selection without training. Default is False.
        experiment_folder (str): Folder to store experiment outputs. Default is "".
        root_folder (str): Root folder for the experiment. Default is "".
        seed (Optional[int]): Random seed for reproducibility. Default is None.
        image_tag (int): Tag for the image version. Default is -1.
        task (str): Task type, e.g., "classification". Default is "classification".
        engine (str): Engine to use, e.g., "pytorch". Default is "pytorch".
    """
    skip_first: bool = False
    resume: Optional[int] = None
    checkpoint_path: str = "/src/checkpoint.model"
    only_select: bool = False
    experiment_folder: str = ""
    root_folder: str = ""
    seed: Optional[int] = None
    image_tag: int = -1
    task: str = "classification"
    engine: str = "pytorch"


@dataclass
class LoggingConfig:
    """
    Configuration for logging and visualization.

    Parameters:
        mlflow_name (str): Name of the MLflow experiment. Default is "streamI".
        store_image (bool): Whether to store images. Default is False.
        store_tsne (bool): Whether to store t-SNE plots. Default is True.
        plot_ind_ood (bool): Whether to plot in-distribution vs out-of-distribution. Default is True.
        verbosity (int): Verbosity level for logging. Default is 0.
    """
    mlflow_name: str = "streamI"
    store_image: bool = False
    store_tsne: bool = True
    plot_ind_ood: bool = True
    verbosity: int = 0


@dataclass
class ActiveLearningScenarioConfig:
    """
    Configuration for the active learning scenario.

    Parameters:
        resume (Optional[int]): Resume from a specific checkpoint or cycle. Default is None.
        query_scenario (str): The query scenario type. Default is "StreamBatch".
        cycles (int): Number of active learning cycles. Default is 10.
        query_size (int): Number of samples to query per cycle. Default is 100.
        is_percent (bool): Whether query_size is a percentage. Default is False.
        full_set (bool): Whether to use the full dataset. Default is False.
        overlap (int): Overlap between cycles. Default is 50.
        over_selection (float): Over-selection factor. Default is 2.0.
        new_model_eval (bool): Whether to evaluate a new model each cycle. Default is False.
        strategy (str): Strategy for model retraining. Default is "retrain".
    """
    resume: Optional[int] = None
    query_scenario: str = "StreamBatch"
    cycles: int = 10
    query_size: int = 100
    is_percent: bool = False
    full_set: bool = False
    overlap: int = 50
    over_selection: float = 2.0
    new_model_eval: bool = False
    strategy: str = "retrain"


@dataclass
class DataScenarioConfig:
    """
    Configuration for data scenario (OpenSet).

    Parameters:
        data_scenario (str): The type of data scenario. Default is "standard".
        ind_classes (List[int]): List of in-distribution class indices. Default is [0, 1, 2, 3, 4, 5].
        near_classes (List[int]): List of near out-of-distribution class indices. Default is [6, 7, 8, 9].
        far_classes (List[int]): List of far out-of-distribution class indices. Default is empty list.
        near_dataset (str): Name of the near out-of-distribution dataset. Default is 'None'.
        far_dataset (str): Name of the far out-of-distribution dataset. Default is 'random'.
        near_ratio (float): Ratio of near out-of-distribution samples. Default is 0.5.
        expanding_th (int): Threshold for expanding classes. Default is 100.
        opensetmode (str): Mode for open set. Default is "InD".
    """
    #OpenSet Config
    data_scenario: str = "standard"
    ind_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    near_classes: List[int] = field(default_factory=lambda: [6, 7, 8, 9])
    far_classes: List[int] = field(default_factory=list)
    near_dataset: str = 'None'
    far_dataset: str = 'random'
    near_ratio: float = 0.5
    expanding_th: int = 100
    opensetmode: str = "InD"


class DataSetConfig(NestedConfig):
    """
    Configuration for a dataset.

    Parameters:
        dataset (str): Name of the dataset. Default is "A2D2".
        order (int): Order or index of the dataset. Default is 1.
        dataset_path (str): Path to the dataset. Default is "".
        size (str): Size of the dataset (e.g., "full"). Default is "full".
        args (Dict): Additional arguments for the dataset. Default is empty dict.
    """
    dataset: str = "cityscapes"
    order: int = 1
    dataset_path: str = "/Datasets/"
    size: str = "full"
    static_configuration: str = "ta"
    data_config: Dict = field(default_factory=lambda: DataConfig())


@dataclass
class DataConfig(NestedConfig):
    # Not used
    dataset: DataSetConfig = field(default_factory=lambda: DataSetConfig())
    datascenario: DataScenarioConfig = field(default_factory=lambda: DataScenarioConfig())


def load_experiment_config():
    experiment_config = asdict(ExperimentConfig())
    experiment_config.update(asdict(LoggingConfig()))
    experiment_config.update(asdict(ActiveLearningScenarioConfig()))
    experiment_config.update(asdict(DataScenarioConfig()))
    experiment_config.update(asdict(DataSetConfig()))
    return experiment_config


def load_training_config():
    training_config = asdict(TrainingConfig())
    training_config.update(asdict(SelectionMethodConfig()))
    training_config.update(asdict(QueryMethodConfig()))
    training_config.update(asdict(ModelConfig()))
    for cls in AbstractSelectionMethodConfig.__subclasses__():
        training_config.update(asdict(cls()))
    return  training_config
