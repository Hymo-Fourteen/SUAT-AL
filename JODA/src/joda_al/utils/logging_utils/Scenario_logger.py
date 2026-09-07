import json
import os.path
import shutil


class ScenarioLogger:
    log_file_name="scenario_log.json"
    state_file_name = "scenario_state.json"
    def __init__(self,experiment_folder,tracking_folder):
        pass

    def save_progress(self,dataset_handler,cycle):
        progress = {}
        progress["cycle"]=cycle
        progress.update(dataset_handler.to_dict())
        with open('data.json', 'w') as f:
            json.dump(progress, f)

    @classmethod
    def save_progress_static(cls,experiment_folder,tracking_folder,dataset_handler,cycle):
        log_file = os.path.join(experiment_folder, tracking_folder, cls.log_file_name)
        state_file = os.path.join(experiment_folder, tracking_folder, cls.state_file_name)

        state = {}
        state["cycle"]=cycle
        state["dataset_handler"] = dataset_handler.to_dict()
        with open(state_file, 'w') as f:
            json.dump(state, f)

        try:
            with open(log_file, 'r') as f:
                progress = json.load(f)
        except FileNotFoundError:
            progress = {}
            progress["cycles"]={}
        progress["cycles"][cycle] = dataset_handler.to_dict()
        with open(log_file, 'w') as f:
            json.dump(progress, f)

    @classmethod
    def get_metadata_static(cls,experiment_folder,tracking_folder):
        state_file = os.path.join(experiment_folder, tracking_folder, cls.state_file_name)
        with open(state_file, 'r') as f:
            progress = json.load(f)
        return progress

    @classmethod
    def get_cycle_static(cls, experiment_folder, tracking_folder):
        progress = cls.get_metadata_static(experiment_folder,tracking_folder)
        return progress["cycle"]

    @classmethod
    def get_dataset_metadata(cls, experiment_folder, tracking_folder):
        progress = cls.get_metadata_static(experiment_folder, tracking_folder)
        return progress["dataset_handler"]

    @staticmethod
    def create_model_cycle_copy(path, verbosity, cycle, model_name=f"checkpoint.model"):
        if verbosity > 2:
            model_path= os.path.join(path, model_name)
            name_components= model_name.split(".")
            cycle_path=os.path.join(path,f"{name_components[0]}_{cycle}.{name_components[1]}")
            shutil.copy2(model_path, cycle_path)