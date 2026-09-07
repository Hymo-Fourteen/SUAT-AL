from typing import Dict, Tuple, List

from joda_al.utils.logging_utils.csv_logger import write_cycle_stats_by_task
from joda_al.Config.config_def import Config
from joda_al.data_loaders.data_handler.data_handler import OpenDataSetHandler
from joda_al.data_loaders.load_dataset import get_training_transformations, get_cls_counts
from joda_al.Pytorch_Extentions.dataset_extentions import TransformSubset
from joda_al.Selection_Methods.method_regestry import LOSS_MODULE_METHODS
from joda_al.model_operations.engine import EngineOperator
from joda_al.model_operations.pytorch.operations import train, EARLY_STOPPING_METRICS, evaluate, train_loss
from joda_al.models.model_utils import setup_model_optimizer_by_strategy, load_model, save_model
from joda_al.utils.logging_utils.training_logger import gl_info
from torch import nn

from joda_al.utils.method_utils import is_loss_learning


class PyTorchOperator(EngineOperator):

    @classmethod
    def train_and_evaluate(cls,
        task_handler,
        device,
        data_handler,
        dataset_config: Dict,
        config: Config,
        experiment_path: str,
        cycle_num: int,
    )-> Tuple[List[float], nn.Module, Dict[str,float], List[Dict[str,float]]]:


        ############################################################
        # Setup
        train_dataset = data_handler.train_pool
        labeled_idx = data_handler.labeled_idcs
        unlabeled_idcs = data_handler.current_unlabeled_idcs
        validation_loader = data_handler.get_val_loader()
        test_loader = data_handler.get_test_loader()
        dataset_config = data_handler.dataset_config
        training_set = data_handler.train_pool
        task = config.experiment_config["task"]
        # Get trainings transformations
        opensetmode = config.experiment_config.get("opensetmode","InD")

        if opensetmode == "InD" and isinstance(data_handler, OpenDataSetHandler):
            training_loader = data_handler.get_train_loader(data_handler.labeled_ind_idcs)
        else:
            training_loader = data_handler.get_train_loader()

        # Update class weights
        if config.training_config["cls_weights"] == "all":
            cls_weights = train_dataset.get_weights(labeled_idx)
        else:
            cls_weights = None
        cls_counts = get_cls_counts(train_dataset, labeled_idx)

        # Setup model for cycle
        method = config.experiment_config["method_type"]

        config.training_config["scheduler"]["max_iter"] = (
            len(training_set) * config.training_config["num_epochs"]
        )
        config.experiment_config["experiment_path"]=experiment_path
        criterion, models, optimizers, schedulers = setup_model_optimizer_by_strategy(
            config.experiment_config,
            dataset_config,
            config.training_config,
            task,
            device,
            method,
            cls_weights,
            cls_counts,
        )

        ############################################################
        # Training and testing
        epoch_loss = config.training_config["epoch_loss"]
        gl_info(epoch_loss)
        # if else/ use pl -> different train
        loss, val_metric = train(
            models,
            method,
            task,
            criterion,
            optimizers,
            schedulers,
            training_loader,
            validation_loader,
            {**config.training_config, **config.experiment_config},
            dataset_config,
            device,
            cycle_num,
        )
        ############################################################
        # Detached Module Training after early stopping and continue training
        if (
            config.training_config.get("train_loss_detached", False) == True
            and method in LOSS_MODULE_METHODS
        ):
            gl_info("Training Loss learning detached")

            criterion, models, optimizers, schedulers, epoch, _ = load_model(
                dataset_config,
                config.training_config,
                task,
                device,
                method,
                experiment_path,
                cls_weights,
                cls_counts,
            )
            epoch_loss = epoch
            lossL, val_accL = train(
                models,
                method,
                task,
                criterion,
                optimizers,
                schedulers,
                training_loader,
                validation_loader,
                {**config.training_config, **config.experiment_config},
                dataset_config,
                device,
                cycle_num,
            )
        ############################################################
        # Training Loss learning Module only
        if (
            config.training_config.get("train_loss", False) == True
            and is_loss_learning(method)
        ):
            gl_info("Training Loss learning only ")
            criterion, models, optimizers, schedulers, epoch, _ = load_model(
                dataset_config,
                config.training_config,
                task,
                device,
                method,
                experiment_path,
                cls_weights,
                cls_counts,
            )
            lossL, val_accL = train_loss(
                models,
                method,
                task,
                criterion,
                optimizers,
                schedulers,
                training_loader,
                validation_loader,
                {**config.training_config, **config.experiment_config},
                dataset_config,
                device,
                experiment_path,
                cycle_num,
                epoch,
            )

        gl_info("Training Done")
        if config.training_config.get("early_stopping", None) in EARLY_STOPPING_METRICS:
            criterion, models, optimizers, schedulers, _, _ = load_model(
                dataset_config,
                config.training_config,
                task,
                device,
                method,
                experiment_path,
                cls_weights,
                cls_counts,
            )
        ############################################################
        # Training and testing
        test_metric, _, _ = evaluate(
            models,
            method,
            task,
            criterion,
            test_loader,
            {**config.training_config, **config.experiment_config},
            dataset_config,
            device,
            cycle_num,
            "test",
        )
        if not config.training_config.get("early_stopping", None) in EARLY_STOPPING_METRICS:

            save_model(config.training_config["num_epochs"], models, optimizers, schedulers, loss, experiment_path)
        write_cycle_stats_by_task(task, loss, val_metric, test_metric, cycle_num)
        return loss, models, test_metric, val_metric

    @classmethod
    def create_model_only(cls, task_handler, device, data_handler, dataset_config: Dict, config: Config, experiment_path: str,
                          cycle_num: int, model_name:str = None, **kwargs):
        train_dataset = data_handler.train_pool
        labeled_idx = data_handler.labeled_idcs
        dataset_config = data_handler.dataset_config
        training_set = TransformSubset(train_dataset, labeled_idx,
                                       **get_training_transformations(config.experiment_config["task"],
                                                                      f'{dataset_config["name"]}-{dataset_config["dataset_scenario"]}'))
        task = config.experiment_config["task"]
        if config.training_config["cls_weights"] == "all":
            cls_weights = train_dataset.get_weights(labeled_idx)
        else:
            cls_weights = None
        cls_counts = get_cls_counts(train_dataset, labeled_idx)

        # Setup model for cycle
        method = config.experiment_config["method_type"]

        config.training_config["scheduler"]["max_iter"] = (
                len(training_set) * config.training_config["num_epochs"]
        )
        config.experiment_config["experiment_path"] = experiment_path
        criterion, models, optimizers, schedulers = setup_model_optimizer_by_strategy(
            config.experiment_config,
            dataset_config,
            config.training_config,
            task,
            device,
            method,
            cls_weights,
            cls_counts,
            model_name,
        )
        return criterion, models, optimizers, schedulers