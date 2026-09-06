from dataclasses import dataclass
from typing import Dict, Union, Optional, Tuple, List
import pandas as pd
import random
import os
import numpy as np
import torch

from joda_al.Selection_Methods.Coverage_Based_Methods.Joda_logic import JodaLogic, SisomLogic, get_layer_output_sizes
from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.data_loaders.sampler import SubsetSequentialSampler
from joda_al.Selection_Methods.Diversity_Based_Methods.diversity_based import CoreSetQuery
from joda_al.Selection_Methods.Uncertainty_Based_Methods.sampling_based import BaldQuery
from joda_al.Selection_Methods.query_method_def import QueryMethod
from torch.utils.data import DataLoader
from joda_al.models.model_lib import ModelHandler
from joda_al.task_supports.pytorch.TaskHandler import TaskHandler
from joda_al.utils.logging_utils.log_writers import gl_info


class JodaQuery(QueryMethod):
    keywords = ["Joda"]
    """
    ActiveCoverageLearning is a query method that selects samples based on coverage-based criteria.

    Args:
        method (str): The name of the method.

    Attributes:
        coverage_method (str): The coverage method used for computing coverage.
        coverage_hyper (dict): The hyperparameters for the coverage method.
        surprise_strategy (str): The strategy for selecting unlabeled samples, can be greedy "surprise" or "coverage".
        sigmoids (dict): The sigmoid values for coverage computation.
        cycle_num (int): The cycle number.
        dataset (str): The name of the dataset.
        scenario (str): The scenario for coverage computation.
        use_pca (bool): Whether to use PCA for coverage computation.
        seperator (str): The separator for coverage computation. (e.g. metric, knn or kmeans)
        sep_metric (str): The metric for coverage computation. (e.g. dista+distb (AB))
        stat_statistics (bool): Whether to compute statistics on the labeled data
        use_gradients (bool): Whether to use gradients for coverage computation.
        num_classes (int): The number of classes in the dataset.
        gain_mode (str): The mode for computing gain, can be standard or energy
        coverage_al (str): The method for computing coverage (individual, incremental, or optimal).

    Methods:
        query: Selects samples based on coverage-based criteria.

    """

    def __init__(self, method, *args, **kwargs):
        # TODO: find data size in arguments, until then hard-coded values of cifar10
        gl_info(f"Method used:\n{method}")

    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs) -> Tuple[List, List]:

        """
        Selects samples based on coverage-based criteria.

        Args:
            models (Dict): The models used for coverage computation.
            handler (Union[TaskHandler, ModelHandler]): The handler for the task or model.
            dataset_handler (DataSetHandler): The handler for the dataset.
            config (Dict): The configuration for training.
            query_size (int): The number of samples to select.
            device: The device to use for computation.
            labeled_idx_set (Optional[int]): The indices of labeled samples.
            unlabeled_idx_set (Optional[int]): The indices of unlabeled samples.

        Returns:
            Tuple[List, List]: The selected samples and their corresponding coverage gains.

        """

        # The following parameters are extracted from the configuration dictionary (`config`) and initialized:
        self.coverage_hyper = config["coverage_hyper"]  # A dictionary of hyperparameters for the coverage method.
        self.surprise_strategy = config["surprise_strategy"]  # Strategy for selecting unlabeled samples (e.g., "surprise" or "coverage").
        self.sigmoids = config["sigmoids"]  # Sigmoid values used for coverage computation.
        self.cycle_num = config["cycle_num"]  # The current cycle number in the active learning process.
        self.dataset = config["dataset"]  # The name of the dataset being used.
        self.scenario = config["data_scenario"]  # The scenario for coverage computation (e.g., data distribution type).
        self.use_pca = config["use_pca"]  # Boolean flag indicating whether PCA is used for dimensionality reduction.
        self.seperator = config["seperator"]  # Method used to separate data points (e.g., metric, knn, or kmeans).
        self.sep_metric = config["sep_metric"]  # Metric used for separation (e.g., distance-based metrics).
        self.stat_statistics = True # Boolean flag to compute statistics on labeled data (always set to True here).
        self.use_balancing = config["acl_balancing"]  # Boolean flag indicating whether balancing is applied in Active Coverage Learning.
        self.apply_ood_filter = config["apply_ood_filter"]  # Boolean flag indicating whether an OOD filter is applied to the data.
        self.num_classes = int(list(models["task"].eval().children())[-1].out_features)  # Number of classes in the dataset, derived from the model's output layer.
        self.device = device  # Specifies the device (e.g., CPU or GPU) used for computation.

        # Normalize dataset names for consistency (e.g., handling variations like "cifar10-variant").
        if "cifar10-" in self.dataset:
            self.dataset = "cifar10"
        if "cifar100-" in self.dataset:
            self.dataset = "cifar100"

        self.gain_mode = config["gain_mode"]  # Mode for computing gain (e.g., "standard", "energy").
        self.coverage_al = config["coverage_al"] # set if coverage should be computed individually, incrementally or optimal

        # input size of image TODO: read from config
        input_size = (1, 3, dataset_handler.dataset_config["size"][0], dataset_handler.dataset_config["size"][1])
        random_data = torch.randn(input_size).to(device=self.device)
        # layer size dict needed for attaching hooks to network
        layer_size_dict = get_layer_output_sizes(models["task"], random_data, mode="nac")

        # create coverage object and compute initial coverage from training data
        self.joda_logic = JodaLogic(models["task"],
                                    layer_size_dict,
                                    hyper=self.coverage_hyper,
                                    cycle_num=self.cycle_num,
                                    dataset=self.dataset,
                                    sigmoids=self.sigmoids,
                                    scenario=self.scenario,
                                    num_class=self.num_classes,
                                    stat_statistics=self.stat_statistics,
                                    use_pca=self.use_pca,
                                    seperator=self.seperator,
                                    sep_metric=self.sep_metric,
                                    use_balancing=self.use_balancing,
                                    apply_ood_filter=self.apply_ood_filter)
        gl_info("Building Coverage..")

        labeled_loader = dataset_handler.get_labeled_pool_loader(labeled_idx_set)
        self.joda_logic.build(labeled_loader)
        gl_info("Computing initial Coverage..")
        gain_list, statistics = self.joda_logic.assess(dataset_handler.get_labeled_pool_loader(labeled_idx_set),
                                                       gain_mode=self.gain_mode)
        if self.stat_statistics:
            statistics_df = pd.DataFrame(statistics.float().tolist(),
                                         columns=["DISTA", "DISTB", "DSA", "ENERGY", "GEN", "sample_index", "OOD"])
            statistics_df["sample_index"] = statistics_df["sample_index"].astype(int)
            statistics_df.to_csv(f"{config['experiment_path']}/stat_statistics_{self.cycle_num}.csv", index=False)
        if self.gain_mode == "energy":
            mean_stats = torch.mean(gain_list, dim=0).float().tolist()
            mean_stats = tuple(mean_stats)
            std_stats = torch.std(gain_list, dim=0).float().tolist()
            std_stats = tuple(std_stats)

            self.joda_logic.weight = mean_stats
            self.joda_logic.std = std_stats
            self.joda_logic.final_std = np.sqrt(2)
            gl_info(f"Gain-Mean: {mean_stats[0]}, Gain-Std: {std_stats[0]}, Cycle: {self.cycle_num}")
        elif self.gain_mode == "pure_energy":
            mean_stats = torch.mean(gain_list, dim=0).float().tolist()
            mean_stats = tuple(mean_stats)
            std_stats = torch.std(gain_list, dim=0).float().tolist()
            std_stats = tuple(std_stats)
            self.joda_logic.weight = (1, 0)
            self.joda_logic.std = (1, 1)
            self.joda_logic.final_std = std_stats[1]
        elif self.gain_mode == "standard":
            mean_stats = torch.mean(gain_list, dim=0).float().tolist()
            mean_stats = tuple(mean_stats)
            std_stats = torch.std(gain_list, dim=0).float().tolist()
            std_stats = tuple(std_stats)
            self.joda_logic.weight = (0, 1)
            self.joda_logic.std = (1, 1)
            self.joda_logic.final_std = std_stats[0]

        unlabeled_loader = dataset_handler.get_unlabeled_pool_loader(unlabeled_idx_set)
        gl_info("Finding top coverage samples..")
        cov_gains, indices, statistics = self.joda_logic.top_k(unlabeled_loader, query_size, self.coverage_al,
                                                               no_label=True, gain_mode=self.gain_mode,
                                                               surprise_strategy=self.surprise_strategy)
        set_indices = [unlabeled_idx_set[i] for i in indices]

        gl_info(f"max cov-gain selected: {cov_gains[0]}")
        gl_info(f"min cov-gain selected: {cov_gains[query_size - 1]}")
        # gl_info(f"mean cov-gain: {np.mean(np.array(cov_gains[:query_size-1]))}")
        gl_info(f"Current Coverage: {self.joda_logic.current}")
        return cov_gains, set_indices


class SisomQuery(QueryMethod):
    keywords = ["SISOM"]
    """
        Naive First Version of ACL for testing, works like this:
            1. Input Init: coverage method [NC, NLC, ...], coverage_hyper [hyperparameter for coverage method]
            2. Input Query: (standard QueryMethod input)
            3. Compute/Get coverage on current training set
            4. Loop through unlabeled pool, compute coverage gain for each sample
            5. sort coverage gains by magnitude (descending)
            6. return sample indices of samples with top query_size coverage gains
    """

    def __init__(self, method, *args, **kwargs):
        # TODO: find data size in arguments, until then hard-coded values of cifar10
        gl_info(f"MODEL:\n{method}")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              training_config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs) -> Tuple[List, List]:

        # this should only be executed for the initial, first query
        torch.cuda.empty_cache()
        # set coverage method and its hyperparameters
        self.coverage_method = training_config["coverage_method"]
        self.coverage_hyper = training_config["coverage_hyper"]
        self.surprise_strategy = training_config["surprise_strategy"]
        self.sigmoids = training_config["sigmoids"]
        self.cycle_num = training_config["cycle_num"]
        self.dataset = training_config["dataset"]
        self.num_classes = int(list(models["task"].eval().children())[-1].out_features)
        if "cifar10-" in self.dataset:
            self.dataset = "cifar10"
        if "cifar100-" in self.dataset:
            self.dataset = "cifar100"
        self.use_shapley = False
        self.gain_mode = training_config["gain_mode"]
        # set if coverage should be computed individually, incrementally or optimal
        self.coverage_al = training_config["coverage_al"]
        # input size of image TODO: read from config
        input_size = (1, 3, 32, 32)
        random_data = torch.randn(input_size).to(device=self.device)
        # layer size dict needed for attaching hooks to network
        layer_size_dict = get_layer_output_sizes(models["task"], random_data, mode="nac")

        # create coverage object and compute initial coverage from training data
        self.sisom_logic: SisomLogic = SisomLogic(models["task"],
                                                  layer_size_dict,
                                                  hyper=self.coverage_hyper,
                                                  cycle_num=self.cycle_num,
                                                  dataset=self.dataset,
                                                  sigmoids=self.sigmoids)
        gl_info("Building Coverage..")
        if self.use_shapley:
            labeled_loader = DataLoader(dataset_handler.train_pool, batch_size=1,
                                        num_workers=training_config["num_workers"],
                                        sampler=SubsetSequentialSampler(labeled_idx_set),
                                        pin_memory=True)
        else:
            labeled_loader = dataset_handler.get_labeled_pool_loader(labeled_idx_set)
        self.sisom_logic.build(labeled_loader)
        self.built = True
        gl_info("Computing initial Coverage..")
        gain_list = self.sisom_logic.assess(dataset_handler.get_labeled_pool_loader(labeled_idx_set),
                                            gain_mode=self.gain_mode)
        if self.gain_mode == "energy":
            mean_stats = torch.mean(gain_list, dim=0).float().tolist()
            mean_stats = tuple(mean_stats)
            std_stats = torch.std(gain_list, dim=0).float().tolist()
            std_stats = tuple(std_stats)

            self.sisom_logic.weight = mean_stats
            self.sisom_logic.std = std_stats
            gl_info(f"Gain-Mean: {mean_stats[0]}, Gain-Std: {std_stats[0]}, Cycle: {self.cycle_num}")
        elif self.gain_mode == "pure_energy":
            mean_stats = torch.mean(gain_list, dim=0).float().tolist()
            mean_stats = tuple(mean_stats)
            std_stats = torch.std(gain_list, dim=0).float().tolist()
            std_stats = tuple(std_stats)
            self.sisom_logic.weight = (1, 0)
            self.sisom_logic.std = std_stats
        elif self.gain_mode == "standard":
            mean_stats = torch.mean(gain_list, dim=0).float().tolist()
            mean_stats = tuple(mean_stats)
            std_stats = torch.std(gain_list, dim=0).float().tolist()
            std_stats = tuple(std_stats)
            self.sisom_logic.weight = (0, 1)
            self.sisom_logic.std = (1, 1)

        unlabeled_loader = dataset_handler.get_unlabeled_pool_loader(unlabeled_idx_set)
        gl_info("Finding top coverage samples..")
        cov_gains, indices = self.sisom_logic.top_k(unlabeled_loader, query_size, self.coverage_al, no_label=True,
                                                    gain_mode=self.gain_mode, surprise_strategy=self.surprise_strategy)
        set_indices = [unlabeled_idx_set[i] for i in indices]

        gl_info(f"max cov-gain selected: {cov_gains[0]}")
        gl_info(f"min cov-gain selected: {cov_gains[query_size - 1]}")
        # logging.info(f"mean cov-gain: {np.mean(np.array(cov_gains[:query_size-1]))}")
        gl_info(f"Current Coverage: {self.sisom_logic.current}")
        return cov_gains, set_indices


class ActiveCoverageLearningAbl(JodaQuery):
    keywords = ["ACLCore", "ACLEntropy","ACLEntropy2","ACLRandom"]

    def __init__(self, method, *args, **kwargs):
        # TODO: find data size in arguments, until then hard-coded values of cifar10
        gl_info(f"MODEL:\n{method}")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.method=method

    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs) -> Tuple[List, List]:
        cov_gains, set_indices=super().query(models, handler, dataset_handler, config, query_size, device, labeled_idx_set, unlabeled_idx_set)
        set_indices2 = [s for (s, v) in zip(set_indices, cov_gains) if v > -10000]
        if self.method == "ACLCore":
            gains, idc = CoreSetQuery("CoreSet").query(models, handler, dataset_handler, config, query_size, device, labeled_idx_set, set_indices2)
        elif self.method == "ACLEntropy":
            gains, idc = BaldQuery("cEnt").query(models, handler, dataset_handler, config, query_size, device, labeled_idx_set, set_indices2)
        elif  self.method == "ACLRandom":
            gains, idc = random.sample(set_indices2, query_size), random.sample(set_indices2, query_size)
        return gains, idc


class NACActiveLearning(QueryMethod):
    keywords = ["NAC"]

    def __init__(self, method, *args, **kwargs):
        # TODO: find data size in arguments, until then hard-coded values of cifar10
        gl_info(f"MODEL:\n{method}")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @dataclass
    class MethodConfig:
        pass

    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs) -> Tuple[List, List]:
        
        try:
            from ood_coverage.openood.postprocessors.nac_postprocessor import NACPostprocessor
        except ImportError:
            raise ImportError("NACPostprocessor not found. Please install the OpenOOD package.")

        from joda_al.Selection_Methods.Coverage_Based_Methods.ood_config_loader import Config, merge_configs
        if not hasattr(self, "coverage_obj"):
            self.coverage_al = config["coverage_al"]
            postprocessor_config_path = os.path.join("./", f'nac_{config["nac_dataset"]}.yml')
            config = Config(postprocessor_config_path)
            config = merge_configs(config,
                        Config(**{'dataset': {
                            'name': config["nac_dataset"]
                        }}))
            self.coverage_obj = NACPostprocessor(config)
            self.coverage_obj.APS_mode = False
            self.coverage_obj.hyperparam_search_done = True
            self.coverage_obj.setup_al(models["task"], dataset_handler.get_labeled_pool_loader(labeled_idx_set))

        unlabeled_loader = dataset_handler.get_unlabeled_pool_loader(unlabeled_idx_set)
        gl_info("Finding top coverage samples..")
        if self.coverage_al == "individual":
            ul_pred, ul_conf, ul_gt = self.coverage_obj.inference(models["task"], unlabeled_loader)
            conf_sort, idx_sort = torch.sort(ul_conf)

        if self.coverage_al == "individual" or self.coverage_al == "incremental":
            gl_info("Updating coverage with selected set..")
            selected_idx_set = idx_sort[:query_size]
            self.coverage_obj.build_nac_al(models["task"], dataset_handler.get_labeled_pool_loader(selected_idx_set))
            
        return conf_sort, idx_sort
