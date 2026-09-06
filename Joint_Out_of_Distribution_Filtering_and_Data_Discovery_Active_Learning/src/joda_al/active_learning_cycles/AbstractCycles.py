import time
from abc import abstractmethod, ABCMeta
from typing import Dict, List, Union, Tuple

import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from joda_al.utils.logging_utils.csv_logger import write_cycle_stats_by_task
from joda_al.data_loaders.load_dataset import get_training_transformations, get_cls_counts
from joda_al.Pytorch_Extentions.dataset_extentions import TransformSubset

from joda_al.Config.config_def import Config
from joda_al.defintions import DataUpdateScenario, SubSubset
from joda_al.task_supports.task_handler_factory import get_task_handler
from joda_al.utils.logging_utils.training_logger import global_write_scalar, global_write_fig, track_selected_images, setup_random
from joda_al.utils.image_creators import create_t_sne_plot
from joda_al.model_operations.operator import Operator
from joda_al.model_operations.pytorch.operations import train, EARLY_STOPPING_METRICS, evaluate
from joda_al.models.model_utils import setup_model_optimizer_loss, load_model


class AbstractActiveLearningCycle():
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def cycle(arg, kwargs):
        ...

    @staticmethod
    @abstractmethod
    def run_scenario(arg, kwargs):
        ...


class ActiveLearningCycle():
    data_creator_function = None

    @classmethod
    def cycledh(cls,
                dh,
                device,
                config,
                query_samples,
                query_size,
                experiment_path,
                cycle_num):

        train_dataset = dh.train_pool
        labeled_idx = dh.labeled_idcs
        unlabeled_idcs = dh.current_unlabeled_idcs
        if config.experiment_config["data_scenario"] == DataUpdateScenario.osal_extending and cycle_num > 0:
            dh.update_data_pool(cls.data_creator_function)
            global_write_scalar("ClassesFound",len(dh.config.experiment_config["ind_classes"]),cycle_num)
        setup_random(seed=config.experiment_config["seed"])
        loss, models, test_metric, val_metric = cls.prepare_model(config, cycle_num, device, dh, experiment_path)

        task = config.experiment_config["task"]
        engine = config.experiment_config["engine"]
        method = config.experiment_config["method_type"]
        config.training_config["cycle_num"] = cycle_num
        config.training_config["dataset"] = config.experiment_config["dataset"]
        start = time.time()
        value_unlabeled_data, selected_unlabeled_idx = query_samples(
            models,
            method,
            dh,
            labeled_idx,
            unlabeled_idcs,
            query_size,
            device,
            {**config.training_config, **config.experiment_config},
            handler=get_task_handler(task, engine),
            cycle_num=cycle_num
        )

        end = time.time()
        global_write_scalar("Selection/Size", len(selected_unlabeled_idx), cycle_num)
        global_write_scalar("Selection/QueryTime", end - start, cycle_num)

        # logging test metrics:
        for metric in test_metric:
            global_write_scalar("Test/%s" %metric, test_metric[metric], cycle_num)

        fig = plt.figure()
        plt.plot(value_unlabeled_data)
        plt.close(fig)
        global_write_fig("Selection/Values", fig, cycle_num)

        if config.experiment_config["store_tsne"]: # TODO pcdet high dim might be a problem for tsne -> use pca first and then tsne
            tsne_plot_selection, tsne_plot_ood = create_t_sne_plot(
                train_dataset,
                labeled_idx,
                unlabeled_idcs,
                selected_unlabeled_idx,
                models,
                config.training_config,
                device,
                handler=get_task_handler(task, engine),
                ds_handler=dh,
                plot_ind_ood=(config.experiment_config["data_scenario"] == DataUpdateScenario.osal or
                              config.experiment_config["data_scenario"] == DataUpdateScenario.osal_extending)
            )
            if (config.experiment_config["data_scenario"] == DataUpdateScenario.osal or
                config.experiment_config["data_scenario"] == DataUpdateScenario.osal_extending):
                global_write_fig("IndOOD/TSNE", tsne_plot_ood, cycle_num)

            # tsne_plot.savefig(
            #     os.path.join(experiment_path, f"tsne{cycle_num}.pdf"), bbox_inches="tight"
            # )
            global_write_fig("Selection/TSNE", tsne_plot_selection, cycle_num)
        # todo add T-SNE
        if config.experiment_config["store_image"]:
            track_selected_images(selected_unlabeled_idx, train_dataset, cycle_num)
        return loss, val_metric, test_metric, value_unlabeled_data, selected_unlabeled_idx

    @classmethod
    def prepare_model(cls, config, cycle_num, device, dh, experiment_path) -> Tuple[List[float], nn.Module, Dict[str,float], List[Dict[str,float]]]:
        operator = Operator(config)
        if config.experiment_config["only_select"]:
            config.experiment_config["experiment_path"] = experiment_path
            criterion, models, optimizers, schedulers = operator.create_model_only(
                device,
                dh,
                dh.dataset_config,
                config,
                experiment_path,
                cycle_num,
            )
            loss = []
            test_metric = []
            val_metric = []
        else:
            loss, models, test_metric, val_metric = operator.train_and_evaluate(
                device,
                dh,
                dh.dataset_config,
                config,
                experiment_path,
                cycle_num,
            )
        return loss, models, test_metric, val_metric

    @classmethod
    @abstractmethod
    def run_scenario(cls,config: Config, device, experiment_path):
        ...

    def setup(self):
        pass

    def new_model_evaluation(
        cls,
        device,
        dataset_config: Dict,
        labeled_idx: List[int],
        train_dataset: Union[Dataset, SubSubset],
        validation_loader: DataLoader,
        test_loader: DataLoader,
        config: Config,
        experiment_path: str,
        cycle_num: int,
    )-> Tuple[List[float], nn.Module, Dict[str,float], List[Dict[str,float]]]:

        # Get trainings transformations
        training_set = TransformSubset(
            train_dataset,
            labeled_idx,
            **get_training_transformations(
                config.experiment_config["task"], f'{dataset_config["name"]}-{dataset_config["dataset_scenario"]}'
            ),
        )
        training_loader = DataLoader(
            training_set,
            batch_size=config.training_config["batch_size"],
            shuffle=True,
            num_workers=config.training_config["num_workers"],
            drop_last=True
        )
        # Update class weights
        if config.training_config["cls_weights"] == "all":
            cls_weights = train_dataset.get_weights(labeled_idx)
        else:
            cls_weights = None
        cls_counts = get_cls_counts(train_dataset, labeled_idx)

        # Setup model for cycle
        method = "Random"
        task = config.experiment_config["task"]
        epoch_loss = config.training_config["epoch_loss"]
        config.training_config["scheduler"]["max_iter"] = (
            len(training_set) * config.training_config["num_epochs"]
        )
        criterion, models, optimizers, schedulers = setup_model_optimizer_loss(
            dataset_config, config.training_config, task, device, method, cls_weights, cls_counts
        )

        # Training and testing
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
            "retrain-eval-checkpoint.model",
        )
        print("Training Done")
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
                "retrain-eval-checkpoint.model",
            )
        test_metric, _, _ = evaluate(
            models,
            method,
            task,
            criterion,
            test_loader,
            config.training_config,
            dataset_config,
            device,
            cycle_num,
            "test",
        )
        write_cycle_stats_by_task(
            task, loss, val_metric, test_metric, cycle_num, only_test=True
        )
        return loss, models, test_metric, val_metric
