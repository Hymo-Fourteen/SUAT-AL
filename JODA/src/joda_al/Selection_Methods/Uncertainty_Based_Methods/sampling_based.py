import collections
from typing import Dict, Union, Optional, Tuple, List
from warnings import warn

import numpy as np
import torch
from torch.utils.data import DataLoader

from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.data_loaders.sampler import SubsetSequentialSampler

from joda_al.Selection_Methods.query_method_def import QueryMethod
from joda_al.Selection_Methods.Uncertainty_Based_Methods.temporal_processing_lib import extract_method, time_processing
from joda_al.models.model_lib import ModelHandler
from joda_al.task_supports.pytorch.TaskHandler import TaskHandler
from joda_al.Selection_Methods.Uncertainty_Based_Methods.batch_bald_utils import get_bald_batch, get_batchbald_batch, get_entropy_batch

from tqdm import tqdm
from joda_al.defintions import Engine

class BaldQuery(QueryMethod):
    batch_bald_keywords = ["BatchBald", "BatchBaldLight", "BatchBaldLightL"]
    log_batch_bald_keywords = ["cBatchBald", "cBatchBaldLight", "cBatchBaldLightL"]
    bald_keywords = ["Bald", "cBald", "BaldDt", "cBaldDt", "cBaldDtn"]
    entropy_keywords = ["cEnt", "cEntDt", "cEntDtn"]
    keywords = batch_bald_keywords+log_batch_bald_keywords+bald_keywords+entropy_keywords

    def __init__(self,method, *args, **kwargs):
        self.method=method
        self.num_mc_samples = 10

    def perform_forwards_passes(self,unlabeled_loader,training_config,model,device):

        # N,K,C = num_passes, num_mc_samples, num_classes
        # Set to eval und activate dropout, as otherwise BatchNorm is active
        model["task"].eval()
        model["task"].set_dropout_status(True)
        with torch.no_grad():
            sample_predictions = []
            for (inputs, _) in unlabeled_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    mc_passes = []
                    for k in range(self.num_mc_samples):
                        scores, _, _ = model["task"](inputs)
                        if training_config["softmax"]:
                            mc_passes.append(scores)
                        else:
                            mc_passes.append(torch.softmax(scores, dim=-1))
                sample_predictions.append(torch.stack(mc_passes, 1))
            log_probs_N_K_C = torch.cat(sample_predictions, 0)
            model["task"].set_dropout_status(False)
        return log_probs_N_K_C

    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:
        unlabeled_idx_set = unlabeled_idx_set if unlabeled_idx_set is not None else dataset_handler.unlabeled_idcs
        unlabeled_loader = dataset_handler.get_unlabeled_pool_loader(unlabeled_idx_set)
        dataset=dataset_handler.train_pool
        self.num_mc_samples = config.get("num_mc_samples", 10)
        log_probs_N_K_C = self.perform_forwards_passes(unlabeled_loader, config, models, device)
        if self.method in ["BatchBald", "BatchBaldLight", "BatchBaldLightL"]:
            canidate_batch = get_batchbald_batch(log_probs_N_K_C, batch_size=query_size,
                                                 num_samples=len(unlabeled_idx_set), device=device)
            set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
            scores = canidate_batch.scores
        elif self.method == "Bald":
            canidate_batch, _ = get_bald_batch(log_probs_N_K_C, batch_size=query_size, device=device)
            set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
            scores = canidate_batch.scores
        elif self.method == "BaldDt":
            _, ent_scores = get_bald_batch(log_probs_N_K_C, batch_size=query_size, device=device)
            uncertainty = np.diff(ent_scores)
            dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
            indices = np.argsort(uncertainty / dt)[::-1]
            set_indices = [unlabeled_idx_set[i] + 1 for i in indices]
            scores = uncertainty / dt
        elif self.method in ["cBatchBald", "cBatchBaldLight", "cBatchBaldLightL"]:
            canidate_batch = get_batchbald_batch(torch.log(log_probs_N_K_C), batch_size=query_size,
                                                 num_samples=len(unlabeled_idx_set), device=device)
            set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
            scores = canidate_batch.scores
        elif self.method == "cBald":
            canidate_batch, _ = get_bald_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device=device)
            set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
            scores = canidate_batch.scores
        elif self.method == "cBaldDt":
            _, ent_scores = get_bald_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device=device)
            uncertainty = np.diff(ent_scores)
            dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
            indices = np.argsort(uncertainty / dt)[::-1]
            set_indices = [unlabeled_idx_set[i] + 1 for i in indices]
            scores = uncertainty / dt
        elif self.method == "cBaldDtn":
            _, ent_scores = get_bald_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device=device)
            uncertainty = np.abs(np.diff(ent_scores))
            dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
            indices = np.argsort(uncertainty / dt)[::-1]
            set_indices = [unlabeled_idx_set[i] + 1 for i in indices]
            scores = uncertainty / dt
        elif self.method == "cEnt":
            canidate_batch, _ = get_entropy_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device=device)
            set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
            scores = canidate_batch.scores
        elif self.method == "cEntDt":
            _, ent_scores = get_entropy_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device=device)
            uncertainty = np.diff(ent_scores)
            dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
            indices = np.argsort(uncertainty / dt)[::-1]
            set_indices = [unlabeled_idx_set[i] + 1 for i in indices]
            scores = uncertainty / dt
        elif self.method == "cEntDtn":
            _, ent_scores = get_entropy_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device=device)
            uncertainty = np.abs(np.diff(ent_scores))
            dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
            indices = np.argsort(uncertainty / dt)[::-1]
            set_indices = [unlabeled_idx_set[i] + 1 for i in indices]
            scores = uncertainty / dt
        else:
            raise NotImplementedError()

        return scores, set_indices


def query_batch_bald(model, dataset, unlabeled_idx_set, device, training_config, query_size, method):
    num_mc_samples=10
    #N,K,C = num_passes, num_mc_samples, num_classes
    unlabeled_loader = DataLoader(dataset, batch_size=training_config["batch_size"],
                                  sampler=SubsetSequentialSampler(unlabeled_idx_set),
                                  pin_memory=True)
    # Set to eval und activate dropout, as otherwise BatchNorm is active
    warn("This is a non Query Method so it could be wrong", DeprecationWarning, stacklevel=2)
    model["task"].eval()
    model["task"].set_dropout_status(True)
    with torch.no_grad():
        sample_predictions=[]
        for (inputs, _) in unlabeled_loader:
            with torch.cuda.device(device):
                inputs = inputs.cuda()
                mc_passes=[]
                for k in range(num_mc_samples):
                    scores, _, _ = model["task"](inputs)
                    if training_config["softmax"]:
                        mc_passes.append(scores)
                    else:
                        mc_passes.append(torch.softmax(scores,dim=-1))
            sample_predictions.append(torch.stack(mc_passes,1))
        log_probs_N_K_C=torch.cat(sample_predictions, 0)
    # note log_probs should be really log values, which was not the case, this is reflected in c for corrected.
    # This is tested with the entropy and mutal infomation implemented in the masters thesis
    if method in ["BatchBald", "BatchBaldLight", "BatchBaldLightL"]:
        canidate_batch = get_batchbald_batch(log_probs_N_K_C, batch_size=query_size, num_samples=len(unlabeled_idx_set),device=device)
        set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
        scores = canidate_batch.scores
    elif method == "Bald":
        canidate_batch,_ = get_bald_batch(log_probs_N_K_C, batch_size=query_size, device = device)
        set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
        scores=canidate_batch.scores
    elif method == "BaldDt":
        _, ent_scores = get_bald_batch(log_probs_N_K_C, batch_size=query_size, device = device)
        uncertainty = np.diff(ent_scores)
        dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
        indices = np.argsort(uncertainty / dt)[::-1]
        set_indices = [unlabeled_idx_set[i]+1 for i in indices]
        scores=uncertainty / dt
    elif method in ["cBatchBald", "cBatchBaldLight", "cBatchBaldLightL"]:
        canidate_batch = get_batchbald_batch(torch.log(log_probs_N_K_C), batch_size=query_size, num_samples=len(unlabeled_idx_set), device = device)
        set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
        scores = canidate_batch.scores
    elif method == "cBald":
        canidate_batch,_ = get_bald_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device = device)
        set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
        scores=canidate_batch.scores
    elif method == "cBaldDt":
        _, ent_scores = get_bald_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device = device)
        uncertainty = np.diff(ent_scores)
        dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
        indices = np.argsort(uncertainty / dt)[::-1]
        set_indices = [unlabeled_idx_set[i]+1 for i in indices]
        scores=uncertainty / dt
    elif method == "cBaldDtn":
        _, ent_scores = get_bald_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device = device)
        uncertainty = np.abs(np.diff(ent_scores))
        dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
        indices = np.argsort(uncertainty / dt)[::-1]
        set_indices = [unlabeled_idx_set[i]+1 for i in indices]
        scores=uncertainty / dt
    elif method == "cEnt":
        canidate_batch, _ = get_entropy_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device=device)
        set_indices = [unlabeled_idx_set[i] for i in canidate_batch.indices]
        scores = canidate_batch.scores
    elif method == "cEntDt":
        _, ent_scores = get_entropy_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device = device)
        uncertainty = np.diff(ent_scores)
        dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
        indices = np.argsort(uncertainty / dt)[::-1]
        set_indices = [unlabeled_idx_set[i]+1 for i in indices]
        scores=uncertainty / dt
    elif method == "cEntDtn":
        _, ent_scores = get_entropy_batch(torch.log(log_probs_N_K_C), batch_size=query_size, device = device)
        uncertainty = np.abs(np.diff(ent_scores))
        dt = np.abs(np.diff(dataset.timestamps[unlabeled_idx_set]))
        indices = np.argsort(uncertainty / dt)[::-1]
        set_indices = [unlabeled_idx_set[i]+1 for i in indices]
        scores=uncertainty / dt
    else:
        raise NotImplementedError()
    model["task"].set_dropout_status(False)
    return scores, set_indices


class BaldLowQuery(BaldQuery):
    """
    Low memory footprint version of the dataset
    """
    batch_bald_keywords = ["BatchBaldLow", "BatchBaldLightLow", "BatchBaldLightLowL"]
    log_batch_bald_keywords = ["cBatchBaldLow", "cBatchBaldLightLow", "cBatchBaldLightLowL"]
    bald_keywords = ["BaldLow", "cBaldLow", "cBaldMaxLow", "BaldDtLow", "cBaldDtLow", "cBaldDtnLow"]
    entropy_keywords = ["cEntLow", "cEntMaxLow", "cEntDtLow", "cEntDtnLow"]
    keywords = batch_bald_keywords + log_batch_bald_keywords + bald_keywords + entropy_keywords

    @staticmethod
    def post_process_idx(method,uncertainty,dataset,unlabeled_idx_set):
        time_process = extract_method(method,["Dt", "Dtn", "Dtp"])
        set_indices = time_processing(time_process, uncertainty, dataset.timestamps, unlabeled_idx_set)
        return set_indices

    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:
        num_mc_samples = self.num_mc_samples
        # N,K,C = num_passes, num_mc_samples, num_classes

        unlabeled_loader = dataset_handler.get_unlabeled_pool_loader(unlabeled_idx_set=unlabeled_idx_set, batch_size=int(config["batch_size"] / 2))

        # Set to eval und activate dropout, as otherwise BatchNorm is active
        log_probs_N_K_C = None
        self.mode = "mean"
        if self.method in ["BatchBald", "BatchBaldLight", "BatchBaldLightL"]:
            unc_func = lambda x: get_batchbald_batch(x, batch_size=query_size, num_samples=len(unlabeled_idx_set),
                                                     device=device)
        elif self.method in ["Bald", "BaldDt"]:
            unc_func = lambda x: get_bald_batch(x, batch_size=query_size, device=device)[1]
        elif self.method in ["Ent", "EntDt", "EntDtn"]:
            unc_func = lambda x: get_bald_batch(x, batch_size=query_size, device=device)[1]
        elif self.method in ["cBatchBald", "cBatchBaldLight", "cBatchBaldLightL"]:
            unc_func = lambda x: get_batchbald_batch(torch.log(x), batch_size=query_size,
                                                     num_samples=len(unlabeled_idx_set), device=device)
        elif self.method in ["cBaldLow", "cBaldMaxLow", "cBaldDtLow", "cBaldDtnLow"]:
            unc_func = lambda x: get_bald_batch(torch.log(x), batch_size=query_size, device=device)[1]
            if self.method in ["cBaldMaxLow"]:
                self.mode="max"
        elif self.method in ["cEntLow", "cEntMaxLow", "cEntDtLow", "cEntDtnLow"]:
            unc_func = lambda x: get_entropy_batch(torch.log(x), batch_size=query_size, device=device)[1]
            if self.method in ["cEntMaxLow"]:
                self.mode="max"

        if config.load_experiment_config()['engine'] == Engine.pl:  # modified function

            models["task"].eval()
            models["task"].to(device)
            models["task"].set_dropout_status(True)

            with torch.no_grad():
                sample_predictions = []
                for inputs in unlabeled_loader:
                    with torch.cuda.device(device):
                        input_images = inputs['image']
                        input_images = input_images.cuda()
                        # mc_passes = torch.tensor([]).cuda()
                        mc_passes = []
                        for k in range(num_mc_samples):
                            out = models["task"](input_images)
                            scores = out["semantic"]
                            mc_passes.append(torch.softmax(scores, dim=1))
                    batch = torch.stack(mc_passes, 1)
                    unc_batch = unc_func(batch)
                    if self.mode == "mean":
                        sample_predictions.append(unc_batch.mean(dim=[1, 2]))
                    elif self.mode == "max":
                        sample_predictions.append(unc_batch.amax(dim=[1, 2]))
                    else:
                        raise NotImplementedError()
                uncertainty = torch.cat(sample_predictions, 0)
            uncertainty = uncertainty.cpu().numpy()
            set_indices = self.post_process_idx(self.method, uncertainty, dataset_handler.train_pool, unlabeled_idx_set)
            # indices = np.argsort(uncertainty)[::-1]
            # set_indices = [unlabeled_idx_set[i] for i in indices] # Removed +1
            scores = uncertainty
            # note log_probs should be really log values, which was not the case, this is reflected in c for corrected.
            # This is tested with the entropy and mutual information implemented in the masters thesis

            models["task"].set_dropout_status(False)
            return scores, set_indices

        else:  # original function

            models["task"].eval()
            models["task"].set_dropout_status(True)

            with torch.no_grad():
                sample_predictions = []
                for (inputs, _) in unlabeled_loader:
                    with torch.cuda.device(device):
                        inputs = inputs.cuda()
                        # mc_passes = torch.tensor([]).cuda()
                        mc_passes = []
                        for k in range(num_mc_samples):
                            scores, _, _ = models["task"](inputs)
                            if isinstance(scores, collections.OrderedDict):
                                scores = scores["out"]
                            if config["softmax"]:
                                mc_passes.append(scores)
                                # mc_passes = torch.cat(mc_passes, scores, 0)
                            else:
                                # mc_passes = torch.cat((mc_passes, torch.softmax(scores, dim=-1)), 1)
                                mc_passes.append(torch.softmax(scores, dim=-1))
                    batch = torch.stack(mc_passes, 1)
                    # unc_batch = torch.cat([unc_func(torch.unsqueeze(b,0)) for b in batch],0)
                    unc_batch = unc_func(batch)
                    if self.mode=="mean":
                        sample_predictions.append(unc_batch.mean(dim=[1, 2]))
                    elif self.mode=="max":
                        sample_predictions.append(unc_batch.amax(dim=[1, 2]))
                    else:
                        raise NotImplementedError()
                uncertainty = torch.cat(sample_predictions, 0)
            uncertainty = uncertainty.cpu().numpy()
            set_indices = self.post_process_idx(self.method, uncertainty, dataset_handler.train_pool, unlabeled_idx_set)
            # indices = np.argsort(uncertainty)[::-1]
            # set_indices = [unlabeled_idx_set[i] for i in indices] # Removed +1
            scores = uncertainty
            # note log_probs should be really log values, which was not the case, this is reflected in c for corrected.
            # This is tested with the entropy and mutual information implemented in the masters thesis

            models["task"].set_dropout_status(False)
            return scores, set_indices


def query_batch_bald_low(model, dataset, unlabeled_idx_set, device, training_config, query_size, method):
    num_mc_samples=10
    #N,K,C = num_passes, num_mc_samples, num_classes
    unlabeled_loader = DataLoader(dataset, batch_size=int(training_config["batch_size"]/2),
                                  sampler=SubsetSequentialSampler(unlabeled_idx_set),
                                  pin_memory=True)
    warn("This is a non Query Method so it could be wrong", DeprecationWarning, stacklevel=2)
    # Set to eval und activate dropout, as otherwise BatchNorm is active
    log_probs_N_K_C=None
    if method in ["BatchBald", "BatchBaldLight", "BatchBaldLightL"]:
        unc_func = lambda x: get_batchbald_batch(x, batch_size=query_size, num_samples=len(unlabeled_idx_set),device=device)
    elif method in ["Bald","BaldDt"]:
        unc_func = lambda x: get_bald_batch(x, batch_size=query_size, device = device)[1]
    elif method in ["Ent","EntDt","EntDtn"]:
        unc_func = lambda x:get_bald_batch(x, batch_size=query_size, device = device)[1]
    elif method in ["cBatchBald", "cBatchBaldLight", "cBatchBaldLightL"]:
        unc_func = lambda x: get_batchbald_batch(torch.log(x), batch_size=query_size, num_samples=len(unlabeled_idx_set), device = device)
    elif method in ["cBald","cBaldDt","cBaldDtn"]:
        unc_func = lambda x: get_bald_batch(torch.log(x), batch_size=query_size, device=device)[1]
    elif method in ["cEntLow","cEntDtLow","cEntDtnLow"]:
        unc_func = lambda x: get_entropy_batch(torch.log(x), batch_size=query_size, device=device)[1]

    model["task"].eval()
    model["task"].set_dropout_status(True)
    with torch.no_grad():
        sample_predictions=[]
        for (inputs, _) in unlabeled_loader:
            with torch.cuda.device(device):
                inputs = inputs.cuda()
                # mc_passes = torch.tensor([]).cuda()
                mc_passes = []
                for k in range(num_mc_samples):
                    scores, _, _ = model["task"](inputs)
                    if isinstance(scores,collections.OrderedDict):
                        scores=scores["out"]
                    if training_config["softmax"]:
                        mc_passes.append(scores)
                        # mc_passes = torch.cat(mc_passes, scores, 0)
                    else:
                        # mc_passes = torch.cat((mc_passes, torch.softmax(scores, dim=-1)), 1)
                        mc_passes.append(torch.softmax(scores, dim=-1))
            batch = torch.stack(mc_passes,1)
            # unc_batch = torch.cat([unc_func(torch.unsqueeze(b,0)) for b in batch],0)
            unc_batch = unc_func(batch)
            sample_predictions.append(unc_batch.mean(dim=[1,2]))
        uncertainty = torch.cat(sample_predictions, 0)
    uncertainty=uncertainty.cpu().numpy()
    indices = np.argsort(uncertainty)[::-1]
    set_indices = [unlabeled_idx_set[i] + 1 for i in indices]
    scores = uncertainty
    # note log_probs should be really log values, which was not the case, this is reflected in c for corrected.
    # This is tested with the entropy and mutal infomation implemented in the masters thesis

    model["task"].set_dropout_status(False)
    return scores, set_indices


class LogitEntropy(QueryMethod):
    keywords=["LogitEntropy"]
    def __init__(self, method, *args, **kwargs):
        self.method = method

    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int]=None,
              unlabeled_idx_set: Optional[int]=None,
              *arg, **kwargs) -> Tuple[List, List]:

        self.device = device
        self.model = models["task"]
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = dataset_handler.get_unlabeled_pool_loader(unlabeled_idx_set)
        entropy, indices = self.get_uncertainty(self.model, unlabeled_loader)

        return entropy, indices

    def get_uncertainty(self, model, unlabeled_loader):
        model.eval()
        entropies = []
        indices = []

        for batch_idx, (data, *_ ) in enumerate(tqdm(unlabeled_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)

            with torch.no_grad():
                scores, _, _ = model(data)
                p = torch.softmax(scores, dim=-1)
                entropy = -torch.sum(p * torch.log(p + 1e-10), dim=1)

            entropies.extend(entropy.cpu().tolist())
            indices.extend(range(batch_idx * unlabeled_loader.batch_size, (batch_idx + 1) * unlabeled_loader.batch_size))

        # Sort the entropies and corresponding indices
        sorted_indices = sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)
        sorted_entropies = [entropies[i] for i in sorted_indices]

        return sorted_entropies, sorted_indices
