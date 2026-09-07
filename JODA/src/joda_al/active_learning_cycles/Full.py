from joda_al.Config.config_def import Config
from joda_al.active_learning_cycles.AbstractCycles import ActiveLearningCycle
from joda_al.data_loaders.data_handler.data_handler_factory import DataSetHandlerFactory
from joda_al.utils.logging_utils.csv_logger import write_csv_file_by_task
from joda_al.data_loaders.dataset_lib import update_means
from joda_al.data_loaders.load_dataset import load_pool_dataset
from joda_al.task_supports.task_handler_factory import get_task_handler
from joda_al.utils.logging_utils.training_logger import write_csv_header, setup_random
from joda_al.model_operations.operator import Operator
from joda_al.utils.logging_utils.log_writers import gl_info


class Full(ActiveLearningCycle):
    data_creator_function = load_pool_dataset
    @classmethod
    def run_scenario(cls, config : Config, device, experiment_path):#
        setup_random(seed=config.experiment_config["seed"])
        total_cycles = config.experiment_config["cycles"]
        task = config.experiment_config["task"]
        engine = config.experiment_config["engine"]
        task_handler = get_task_handler(task, engine)
        stats = task_handler.get_stats(config.experiment_config["data_scenario"])
        write_csv_header(experiment_path, "stats_file", stats)

        dh = DataSetHandlerFactory(config)(*cls.data_creator_function(config),config)
        validation_loader = dh.get_val_loader()
        test_loader = dh.get_test_loader()

        training_pool = dh.train_pool
        label_idx = dh.labeled_idcs
        unlabeled_idx = dh.unlabeled_idcs
        dataset_config = dh.dataset_config
        all_idx = label_idx + unlabeled_idx
        dh.labeled_idcs = all_idx
        gl_info(f"Fulltraining:")
        update_means(
            config.experiment_config["dataset"], all_idx, training_pool, dh.validation_set, dh.test_set
        )
        setup_random(seed=config.experiment_config["seed"])
        # Final model and results
        operator = Operator(config)
        loss, models, test_metric, val_metric = operator.train_and_evaluate(device,
                                                                            dh,
                                                                            dataset_config,
                                                                            config,
                                                                            experiment_path,
                                                                            total_cycles)
        write_csv_file_by_task(
            experiment_path,
            config.experiment_config["task"],
            loss,
            test_metric,
            val_metric,
            value_unlabeled_data=None,
            selected_unlabeled_idx=None,
        )
        # Update the labeled dataset and the unlabeled dataset, respectively