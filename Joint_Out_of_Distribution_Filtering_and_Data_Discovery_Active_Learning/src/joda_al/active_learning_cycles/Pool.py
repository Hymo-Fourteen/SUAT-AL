from joda_al.active_learning_cycles.AbstractCycles import ActiveLearningCycle
from joda_al.utils.logging_utils.csv_logger import write_csv_file_by_task
from joda_al.Config.config_def import Config
from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.data_loaders.data_handler.data_handler_factory import DataSetHandlerFactory
from joda_al.data_loaders.dataset_lib import update_means
from joda_al.data_loaders.load_dataset import load_pool_dataset
from joda_al.Selection_Methods.sample_selection import factory_query
from joda_al.task_supports.task_handler_factory import get_task_handler
from joda_al.utils.logging_utils.Scenario_logger import ScenarioLogger
from joda_al.utils.logging_utils.log_writers import gl_info
from joda_al.utils.logging_utils.training_logger import write_csv_header, global_write_scalar, setup_random
from joda_al.model_operations.operator import Operator


class PoolCycle(ActiveLearningCycle):
    data_creator_function = load_pool_dataset
    @classmethod
    def run_scenario(cls, config : Config, device, experiment_path):
        # Init Random and variables
        setup_random(seed=config.experiment_config["seed"])
        total_cycles = config.experiment_config["cycles"]
        query_size = config.experiment_config["query_size"]
        is_percent = config.experiment_config["is_percent"]
        task = config.experiment_config["task"]
        engine = config.experiment_config["engine"]
        th = get_task_handler(task, engine)

        # Create DataSetHandler
        dh : DataSetHandler = DataSetHandlerFactory(config)(*cls.data_creator_function(config), config)

        validation_loader = dh.get_val_loader()
        test_loader = dh.get_test_loader()
        training_pool = dh.train_pool
        label_idx = dh.labeled_idcs
        unlabeled_idcs = dh.unlabeled_idcs
        dh.current_unlabeled_idcs = dh.unlabeled_idcs

        query_samples = factory_query

        # Start logging
        stats = th.get_stats(config.experiment_config["data_scenario"])
        write_csv_header(experiment_path, "stats_file", stats)

        # Define Query Size
        if is_percent:
            query_size_s = int((len(unlabeled_idcs) + len(label_idx)) * (query_size / 100.0))
        else:
            query_size_s = query_size

        # Check if experiment is resumed
        if config.experiment_config["resume"] is not None and config.experiment_config["resume"] > 0:
            initial_cycle = ScenarioLogger.get_cycle_static(experiment_path,"")
            dh.load_from_dict(ScenarioLogger.get_dataset_metadata(experiment_path, ""))
        else:
            initial_cycle = 0

        # Start the active learning cycles
        for cycle in range(initial_cycle,total_cycles):
            gl_info(f"Cycles: {cycle}/{total_cycles}")
            update_means(
                config.experiment_config["dataset"],
                label_idx,
                training_pool,
                dh.validation_set,
                dh.test_set,
            )
            global_write_scalar("Selection/TotalSize", len(label_idx), cycle)
            setup_random(seed=config.experiment_config["seed"])
            (
                loss,
                val_metric,
                test_metric,
                value_unlabeled_data,
                selected_unlabeled_idx,
            ) = cls.cycledh(
                dh,
                device,
                config,
                query_samples,
                query_size_s,
                experiment_path,
                cycle,
            )

            # Retrain a fresh model for evaluation
            if config.experiment_config["new_model_eval"]:
                setup_random(seed=config.experiment_config["seed"])
                _, _, test_metric, _ = cls.new_model_evaluation(
                    device,
                    dh.dataset_config,
                    label_idx,
                    training_pool,
                    validation_loader,
                    test_loader,
                    config,
                    experiment_path,
                    cycle,
                )

            # Update the labeled dataset and the unlabeled dataset, respectively
            new_training_idx = label_idx + selected_unlabeled_idx
            unlabeled_idx=dh.current_unlabeled_idcs
            new_unlabeled_idx = [
                idx for idx in unlabeled_idx if idx not in selected_unlabeled_idx
            ]
            label_idx = new_training_idx
            dh.update_labeled_idx_set(label_idx)
            unlabeled_idx = new_unlabeled_idx
            dh.update_current_unlabeled_idx_set(unlabeled_idx)

            # Save the current state of the experiment
            if config.experiment_config["verbosity"] > 0:
                ScenarioLogger.save_progress_static(experiment_path,"",dh,cycle)
                ScenarioLogger.create_model_cycle_copy(experiment_path,config.experiment_config["verbosity"],cycle)
            # Log the results
            write_csv_file_by_task(
                experiment_path,
                config.experiment_config["task"],
                loss,
                test_metric,
                val_metric,
                value_unlabeled_data=list(value_unlabeled_data),
                selected_unlabeled_idx=selected_unlabeled_idx,
            )
            if len(unlabeled_idx) == 0:
                gl_info("Unlabeled Set Empty")
                break

        # Last round no selection
        global_write_scalar("Selection/TotalSize", len(label_idx), total_cycles)
        setup_random(seed=config.experiment_config["seed"])

        # Final model and results
        operator = Operator(config)
        loss, models, test_metric, val_metric = operator.train_and_evaluate(device,
            dh,
            dh.dataset_config,
            config,
            experiment_path,
            total_cycles)
        for metric in test_metric:
            global_write_scalar("Test/%s" %metric, test_metric[metric], total_cycles)
        for metric in test_metric:
            global_write_scalar("FinalTest/%s" %metric, test_metric[metric], total_cycles)

        if config.experiment_config["new_model_eval"]:
            setup_random(seed=config.experiment_config["seed"])
            _, _, test_metric, _ = cls.new_model_evaluation(
                device,
                dh.dataset_config,
                label_idx,
                training_pool,
                validation_loader,
                test_loader,
                config,
                experiment_path,
                total_cycles,
            )
        write_csv_file_by_task(
            experiment_path,
            config.experiment_config["task"],
            loss,
            test_metric,
            val_metric,
            value_unlabeled_data=list(),
            selected_unlabeled_idx=list(),
        )