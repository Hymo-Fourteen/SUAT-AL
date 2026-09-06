import importlib
import sys



import argparse
import json
import os
import warnings
from dataclasses import dataclass, is_dataclass, fields

import yaml
from typing import Dict, Any, _GenericAlias
from typing_extensions import get_origin
from joda_al.Config.config import (
    create_argparser,
    load_experiment_config,
    update_dict,
    load_training_config,
    StoreBooleanAction,
    TRUE, FALSE
)
from joda_al.Config.config_def import Config, NestedConfig
from joda_al.Selection_Methods.SelectionMethodFactory import check_existence

from joda_al.data_loaders.dataset_registry import ALL_DATASETS


def get_default_config(path): #used by Julius
    """returns the DefaultConfig contained in the same directory as the path file. The DefaultConfig class must be in a file called default_config.py

    Args:
        path (str): yaml config file

    Returns:
        class DefaultConfig: Default Configuration for yaml file
    """

    # Get the directory of the current file
    current_file_directory = os.path.dirname(os.path.abspath(path))

    # Construct the full path to the default_config.py file
    default_config_path = os.path.join(current_file_directory, "default_config.py").replace("yaml", "dataclass")

    # Load the module from the file path
    spec = importlib.util.spec_from_file_location("default_config", default_config_path)
    default_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(default_config_module)

    # Access the DefaultConfig class
    DefaultConfig = default_config_module.DefaultConfig
    return DefaultConfig

    


def create_file_argument_parser():
    #used by Julius
    parser = argparse.ArgumentParser()
    parser.add_argument(
        f"--config-path",
        type=str,
        required=False,
        help="Optional config loading interface, needs to be either yaml or json",
        nargs='*'
    )
    return parser


def create_serialized_config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--config-string", type=str, required=False, help="Optional serialized config string as json")
    return parser


def parse_config() -> Config:
    file_args, unknown_args = create_file_argument_parser().parse_known_args()
    if file_args.config_path is not None:
        return parse_file_cli_hierarchy(file_args.config_path)
    serialized_args, unknown_args = create_serialized_config_parser().parse_known_args()
    if serialized_args.config_string is not None:
        return parse_from_serialized(serialized_args.config_string)
    args, unknown_args = create_argparser().parse_known_args()
    return parse_from_cli(args, unknown_args)


def parse_and_validate_config():
    config = parse_config()
    experiment_config, training_config = (
        config["experiment_config"],
        config["training_config"],
    )
    method_validation(experiment_config["method_type"])
    assert (
        experiment_config["dataset"] in ALL_DATASETS
    ), "No dataset %s! Try options %s" % (experiment_config["dataset"], ALL_DATASETS)
    print("Dataset: %s" % experiment_config["dataset"])
    print("Method type:%s" % experiment_config["method_type"])
    print("Cycle type:%s" % experiment_config["base"])
    return experiment_config, training_config


def parse_from_cli(args, unknown_args):

    arg_config = parse_arg_config(args, unknown_args,filter_None=True)
    experiment_config = update_dict(load_experiment_config(), arg_config)
    training_config = update_dict(load_training_config(), arg_config)
    return Config(experiment_config, training_config)


def parse_arg_config(args, unknown_args, filter_None=False): #used by Julius
    if len(unknown_args) > 0:
        warnings.warn(f"{unknown_args} could not be parsed")
    parsed_config = vars(args)
    if filter_None:
        parsed_config = {k: v for k, v in parsed_config.items() if v is not None}

    #Update Dynamic Defaults
    # if parsed_config["base"] == "Full":
    #     parsed_config["method_type"] = "Full"
    # Override experiment folder on cluster
    if (
        parsed_config.get("experiment_folder", None) is None
        or parsed_config.get("experiment_folder") == ""
    ):
        parsed_config["experiment_folder"] = os.getenv("MY_POD_NAME","" )
    # Nested experiment read
    config_dict = {}
    for key, value in parsed_config.items():
        if "." in key:
            keys = key.split(".")
            current = config_dict
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        else:
            config_dict[key] = value
    return config_dict


def parse_from_file(config_file):
    parsed_config = parse_from_file_no_defaults(config_file)
    parsed_experiment_config = parsed_config.get("experiment_config", {})
    parsed_training_config = parsed_config.get("training_config", {})
    if (
        parsed_experiment_config["experiment_folder"] is None
        or parsed_experiment_config["experiment_folder"] == ""
    ):
        parsed_experiment_config["experiment_folder"] = os.getenv("MY_POD_NAME", None)
    parsed_training_config = {
        k: v for k, v in parsed_training_config.items() if v is not None
    }
    parsed_experiment_config = {
        k: v for k, v in parsed_experiment_config.items() if v is not None
    }
    experiment_config = update_dict(load_experiment_config(), parsed_experiment_config)
    training_config = update_dict(load_training_config(), parsed_training_config)
    return experiment_config, training_config


def parse_from_files_no_defaults(config_file_str):
    configs = [parse_from_file_no_defaults(config_file) for config_file in config_file_str]
    ret_config={}
    for conf in configs:
        ret_config = update_dict(ret_config,conf,expand=True)
    return ret_config


def parse_from_file_no_defaults(config_file):
    with open(config_file, "r") as f:
        if "yaml" in config_file or "yml" in config_file:
            parsed_config = yaml.safe_load(f)
        elif "json" in config_file:
            parsed_config = json.load(f)
        else:
            raise NotImplementedError("Config file type not supported")
    return parsed_config


def parse_file_cli_hierarchy(config_file):
    parsed_config = parse_from_files_no_defaults(config_file)
    args, unknown_args = create_argparser(
        remove_default_values=True, defaults_to_keep=["experiment_folder"]
    ).parse_known_args()
    idx = unknown_args.index("--config-path")
    del unknown_args[idx]  # del --config-path
    del unknown_args[idx]  # del Argument
    arg_config = parse_arg_config(args, unknown_args, filter_None=True)
    # Update Defaults with File Config
    parsed_experiment_config = parsed_config.get("experiment_config", {})
    parsed_training_config = parsed_config.get("training_config", {})
    experiment_config = update_dict(load_experiment_config(), parsed_experiment_config)
    training_config = update_dict(load_training_config(), parsed_training_config)
    # Update Merged config with cli config
    experiment_config = update_dict(experiment_config, arg_config)
    training_config = update_dict(training_config, arg_config)

    return Config(experiment_config, training_config)


def parse_from_serialized(config_string):
    """
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL'
    """

    # Convert Python String to proper json
    config_string = (
        config_string.replace("None", "null")
        .replace("False", "false")
        .replace("True", "true")
        .replace("'", '"')
    )
    parsed_config = json.loads(config_string)
    # Update specific config
    parsed_experiment_config = parsed_config["experiment_config"]
    parsed_training_config = parsed_config["training_config"]
    if (
        parsed_experiment_config["experiment_folder"] is None
        or parsed_experiment_config["experiment_folder"] == ""
    ):
        parsed_experiment_config["experiment_folder"] = os.getenv("MY_POD_NAME", None)
    parsed_training_config = {
        k: v for k, v in parsed_training_config.items() if v is not None
    }
    parsed_experiment_config = {
        k: v for k, v in parsed_experiment_config.items() if v is not None
    }
    experiment_config = update_dict(load_experiment_config(), parsed_experiment_config)
    training_config = update_dict(load_training_config(), parsed_training_config)
    return Config(experiment_config, training_config)

def parse_all_args():
    # args = create_argparser().parse_args()
    args, unknown_args =create_argparser().parse_known_args()
    method_validation(args.method_type)
    dataset_validation(args)


def dataset_validation(args):
    assert args.dataset in ALL_DATASETS, 'No dataset %s! Try options %s' % (args.dataset, ALL_DATASETS)


def add_dataclass_to_args(parser, dataclass_obj, prefix=""): #used by Julius
    for field in fields(dataclass_obj):
        if is_dataclass(field.type):
            # if is_dataclass(getattr(dataclass_obj, field.name)):
            add_dataclass_to_args(
                parser,
                # getattr(dataclass_obj, field.name),
                field.type,
                prefix=f"{prefix}{field.name}.",
            )
        elif get_origin(field.type) is dict:
            parser.add_argument(
                f"--{prefix}{field.name}",
                type=json.loads,
                help=f"Description for {field.name}",
            )
        elif isinstance(field.type, _GenericAlias):
            parser.add_argument(
                f"--{prefix}{field.name}",
                type=json.loads,
                help=f"Description for {field.name}",
            )
        elif field.type == bool:
            parser.add_argument(
                f"--{prefix}{field.name}",
                choices=(TRUE, FALSE),
                action=StoreBooleanAction,
                help=f"Description for {field.name}",
            )
        else:
            parser.add_argument(
                f"--{prefix}{field.name}",
                type=field.type,
                help=f"Description for {field.name}",
            )
    return parser


def parse_nested_args(namespace):
    nested_args = {}
    for key, value in vars(namespace).items():
        if "." in key:
            keys = key.split(".")
            d = nested_args
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
        else:
            nested_args[key] = value
    return nested_args


def method_validation(method_type):
    if "->" in method_type:
        methods = method_type.split("->")
        for method in methods:
            assert check_existence(method), "No method %s!" % (
                method,
            )
    else:
        assert check_existence(method_type), "No method %s!" % (
            method_type,

        )
    return True


def dataclass_to_dict(obj: Any) -> Dict[str, Any]: #used by Julius
    """
    Recursively converts a nested dataclass structure into a dictionary.
    Args:
        obj (Any): The object to convert.
    Returns:
        Dict[str, Any]: The dictionary representation of the input object.
    """
    if is_dataclass(obj):
        return {
            field.name: dataclass_to_dict(getattr(obj, field.name))
            for field in fields(obj)
        }
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj