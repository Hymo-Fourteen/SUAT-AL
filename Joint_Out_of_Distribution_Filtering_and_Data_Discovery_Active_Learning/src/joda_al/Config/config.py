import argparse
import copy
import json
import os
from datetime import datetime

from typing import List

from joda_al.Config.config_dataclass import load_training_config as load_training_config_dc, load_experiment_config as load_experiment_config_dc
from joda_al.utils.logging_utils.training_logger import (
    snake_case_to_minus_case,
    TF_BOARD,
)


def get_default_experiment_path(config):
    if config.experiment_config["experiment_folder"] == '' or config.experiment_config["experiment_folder"] is None:
        default_experiment_folder = f'{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}_{config.experiment_config["method_type"]}_{config.experiment_config["seed"]}'
        return  default_experiment_folder


def update_dict(master, updated, expand=False):
    # master.update({k: updated[k] for k, v in master.items() if k in updated})
    merged = copy.deepcopy(master)
    for key, value in updated.items():
        if key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key]=update_dict(merged[key], value, expand=True)
            else:
                merged[key] = value
        else:
            if expand:
                merged[key] = value
    return merged


def load_experiment_config():
    return load_experiment_config_dc()


def load_training_config():
    return load_training_config_dc()


TRUE, FALSE = "True", "False"

data_types = {"lr": float, "lr_backbone": float, "gcn_lr": float, "momentum": float, "wdcay": float, "margin": float,
         "batch_size": int, "num_epochs": int, "lloss_weight": float, "epoch_loss": int, "softmax": bool,
         "optimizer": str, "method_type": str, "dataset": str, "num_workers": int, "full_set": bool,
         "data_scenario": str, "near_dataset": str, "far_dataset": str, "ind_classes": json.loads, "near_classes": json.loads, "far_classes": json.loads, "expanding_th": int,
         "dataset_path": str,"early_stopping": str, "patience": int, "is_percent": bool,
         "cycles": int, "query_size": int, "experiment_folder": str, "store_image": bool,
         "root_folder": str, "epochs_vaal": int, "seed": int, "cls_weights": str, "model": str,
         "coverage_method": str, "coverage_hyper": float, "coverage_al": str, "surprise_strategy": str, "gain_mode": str, "use_gradients": bool, "use_pca": bool, "seperator": str, "sep_metric": str,
         "nac_config": str, "nac_dataset": str, "nnguide_config": str, "sigmoids": json.loads, "layer_selection": json.loads, "osr": bool, "acl_balancing": bool, "apply_ood_filter":  bool,"lambda_oe": float,
         "skip_first": bool, "checkpoint_path": str,
         "epoch_start": int,
         "near-ratio": float,
         "num_mc_samples": int,
         "ensemble": bool,
         "ensemble_size": int,
         "ensemble_training": bool,
         "eval_frequencey": int,
         "verbosity": int,
         "resume":int,
         "gcn_dropout_rate": float, "gcn_lambda_loss": float, "gcn_hidden_units": float, "gcn_s_margin": float,
         "gcn_graph_type": str,
         "order": int,
         "image_tag": str,
         "prob_cover_delta": int,
         "load_from_checkpoint": str,
         # Visu
         "store_predictions": bool, "store_tsne": bool, "plot_ind_ood": bool,
         # Al Scenario Settings:
         "strategy": str,
         "base": str, "over_selection": float,  "overlap": int,
         "opensetmode": str,
         # Scheduler
         "milestones": json.loads, "scheduler": json.loads, "warm_up": bool,
         # Loss learning:
         "head_loss_features": bool, "train_loss": bool, "train_loss_detached": bool,  "lloss_reg": float, "loss_module_lr": float,
         "new_model_eval": bool,
         #  "lower_loss": bool, "higher_loss": bool, "higher_loss2": bool,
         # Model
         "pretrained": bool,
         "pretrained_backbone": bool,
         "aux_loss": bool,
         # Experiment config
         "task": str,
         "engine": str,
         "only_select": bool,
         # Submodular optimization
         "epsilon": float,
         "do_ratio": float,
         #Loss
         "loss_func": str, "alpha": float, "triplet_margin": float, "triplet_mode": str,
         # Flow
         "num_flow_warmup": int, "num_flow_finetune": int, "flow_cross_entropy_weight": float,
         "flow_type": str, "flow_num_layers": int, "flow_early_stopping": bool, "flow_all_data": bool,
         #MLflow
         "mlflow_name": str,
         "agg_mode": str,
         }


def extract_config_types(config_cls):
    """
    Extracts field names and types from a dataclass config definition.

    Args:
        config_cls: The dataclass type (not instance).

    Returns:
        Dict[str, type]: Mapping from field name to type.
    """
    from dataclasses import fields
    return {f.name: f.type for f in fields(config_cls)}


def remove_defauts(params, keep :List[str]=None):
    if keep is None:
        keep=[]
    return {k: None if k not in keep else v for k, v in params.items()}


class StoreBooleanAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        assert values in (TRUE, FALSE)
        setattr(namespace, self.dest, values == TRUE)


def create_argparser(
    remove_default_values=False, defaults_to_keep=None, types=None):
    if defaults_to_keep is None:
        defaults_to_keep = []
    if types is None:
        types = data_types

    if remove_default_values:
        experiment_config = remove_defauts(load_experiment_config(), defaults_to_keep)
    else:
        experiment_config=load_experiment_config()
    types_s = snake_case_to_minus_case(types)
    parser = argparse.ArgumentParser()
    for parameter_dict in [remove_defauts(load_training_config(),defaults_to_keep), experiment_config]:
        parameter_dict = snake_case_to_minus_case(parameter_dict)
        for k, al in parameter_dict.items():
            if types_s[k] is bool:
                parser.add_argument(
                    f"--{k}",
                    choices=(TRUE, FALSE),
                    action=StoreBooleanAction,
                )
            else:
                parser.add_argument(
                    f"--{k}", default=al, type=types_s[k]
                )
    return parser


def mark_as_failed(root_path, experiment_path):
    logging_folder = os.path.join(root_path, TF_BOARD, experiment_path)
    os.rename(logging_folder, f"{logging_folder}-failed")
    return f"{logging_folder}-failed"
