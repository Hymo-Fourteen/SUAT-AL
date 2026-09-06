# Python
# Custom
import os
import torch
import traceback
#from dotenv import load_dotenv

#load_dotenv()

try:
    from KECOR.pcdet_al_wrapper import patches
except ImportError:
    print("Warning: Could not import 'patches' for 3DDet. Some functionality may be unavailable.")


from joda_al.utils.logging_utils.log_writers import init_global_logger, deinit_global_logger, gl_error, gl_info, \
    log_pretty
from joda_al.active_learning_cycles.Full import Full
from joda_al.active_learning_cycles.Pool import PoolCycle
from joda_al.Config.config_def import Config
from joda_al.Config.parser_io import parse_and_validate_config
from joda_al.defintions import ActiveLearningScenario, Scenario
from joda_al.Config.config import mark_as_failed
from joda_al.utils.logging_utils.training_logger import write_json, init_tfboard_writer, global_write_text, \
    global_writer_mark_failed, global_log_file, init_file_writer


# args = parser.parse_args()
CUDA_VISIBLE_DEVICES = 0
DEBUG = False


def start_experiments(experiment_config, training_config):

    config = Config(experiment_config, training_config)
    if config.experiment_config["query_scenario"] == ActiveLearningScenario.pool:
        cycle_func = PoolCycle.run_scenario #run_pool
    elif config.experiment_config["query_scenario"] == ActiveLearningScenario.full:
        cycle_func = Full.run_scenario #run_full
    else:
        raise NotImplementedError("Current Cycle is not implemented")

    device = CUDA_VISIBLE_DEVICES
    device = f"cuda:{device}" if device != "" else "cpu"
    device = torch.device(device)

    gl_info(device)

    dataset = {}
    dataset["name"] = experiment_config["dataset"]
    dataset["path"] = experiment_config["dataset_path"]
    root_folder = experiment_config["root_folder"]
    experiment_path = os.path.join(root_folder, experiment_config["experiment_folder"])

    # fix seed if not -1 or None
    config.training_config["experiment_path"] = experiment_path
    full_config = config.experiment_config.copy()
    full_config.update(config.training_config)
    os.makedirs(experiment_path, exist_ok=True)
    full_config2=full_config.copy()

    # Write Config
    write_json(experiment_path, "arg_config.json", full_config2)
    config.experiment_config["experiment_path"] = experiment_path
    init_global_logger(root_folder, config.experiment_config["experiment_folder"], use_fh=True)

    init_tfboard_writer(root_folder, config.experiment_config["experiment_folder"])
    init_file_writer(root_folder, config.experiment_config["experiment_folder"])

    config.training_config["seed"] = config.experiment_config["seed"]
    gl_info(full_config2)
    gl_info(f"Config is: {log_pretty(full_config2)}")

    # Run Cycle Function
    if DEBUG:
        cycle_func(config, device, experiment_path)
        deinit_global_logger()
    else:
        try:
            cycle_func(config, device, experiment_path)
        except Exception as e:
            os.rename(experiment_path, f"{experiment_path}-failed")
            global_write_text(f"Got error: {str(e)} with trace {traceback.format_exc()}","Errorlog")
            gl_error(f"Got error: {str(e)} with trace {traceback.format_exc()}")
            global_writer_mark_failed()
            mark_as_failed(root_folder, experiment_config["experiment_folder"])
            experiment_path = f"{experiment_path}-failed"
            raise e
        finally:
            deinit_global_logger()
            global_log_file(os.path.join(experiment_path, 'log.txt'))


if __name__ == '__main__':
    experiment_config, training_config = parse_and_validate_config()
    start_experiments(experiment_config, training_config)