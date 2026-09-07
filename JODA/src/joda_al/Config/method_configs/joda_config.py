from dataclasses import dataclass, field

from typing import Dict, Optional
from joda_al.Config.method_configs.query_config import QueryMethodConfig


@dataclass
class JodaMethodConfig(QueryMethodConfig):
    method_type: str = "lloss"
    method: str = "Joda"
    args: Dict = field(default_factory=lambda: JodaArgsConfig())

@dataclass
class JodaArgsConfig:
    use_acl_balancing: bool = True
    use_gradients: bool = True
    use_pca: bool = False


@dataclass
class AbstractSelectionMethodConfig:
    pass


@dataclass
class VAALConfig(AbstractSelectionMethodConfig):
    epochs_vaal: int = 100


@dataclass
class GCNConfig(AbstractSelectionMethodConfig):
    gcn_dropout_rate: float = 0.3
    gcn_lambda_loss: float = 1.2
    gcn_hidden_units: int = 128
    gcn_s_margin: float = 0.1
    gcn_lr: float = 1e-3
    gcn_graph_type: str = "gcn"


@dataclass
class LossLearningConfig(AbstractSelectionMethodConfig):
    train_loss_detached: bool = False
    loss_module_lr: float = -1
    epoch_loss: int = 160
    train_loss: bool = False
    head_loss_features: bool = False
    lloss_reg: float = 0.5
    margin: float = 1.0
    lloss_weight: float = 1.0
    lloss_loss_function: str = "mse"  # "mse", "l1", "bce", "ce"


@dataclass
class CoverageConfig(AbstractSelectionMethodConfig):
    coverage_method: str = "NC"
    coverage_hyper: float = 0.75
    coverage_al: str = "incremental"
    surprise_strategy: str = "surprise"
    gain_mode: str = "standard"
    use_gradients: bool = True
    acl_balancing: bool = True
    use_pca: bool = False
    seperator: str = "metric"
    sep_metric: str = "AB"
    layer_selection: Optional[str] = ""
    nac_config: str = "./src/joda_al/Config/Selection_Methods/nac/resnet/"
    nac_dataset: str = "cifar10"
    sigmoids: Dict = field(default_factory=lambda: {"AdaptiveAvgPool2d-1": 1, "Sequential-3": 1, "Sequential-2": 1, "Sequential-1": 1})
    apply_ood_filter: bool = True
    lambda_oe: float = 0.5


@dataclass
class ProbCoverConfig(AbstractSelectionMethodConfig):
    prob_cover_delta: float = 0.25


@dataclass
class FlowConfig(AbstractSelectionMethodConfig):
    num_flow_warmup: int = 5
    num_flow_finetune: int = 5
    flow_cross_entropy_weight: float = 0.0
    flow_type: str = "radial"
    flow_num_layers: int = 5
    flow_early_stopping: bool = False
    flow_all_data: bool = False


@dataclass
class SelectionMethodConfig:
    agg_mode: str = "mean"
    num_mc_samples: int =10
