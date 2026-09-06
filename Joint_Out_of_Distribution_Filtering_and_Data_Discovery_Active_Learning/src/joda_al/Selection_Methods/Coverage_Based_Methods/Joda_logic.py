import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from joda_al.defintions import DataUpdateScenario
from joda_al.utils.logging_utils.log_writers import gl_info


class JodaLogic():

    def __init__(self, model, layer_size_dict, cycle_num=0, hyper=None, dataset="cifar10", scenario="standard", min_var=1e-5, num_class=10, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.dataset = dataset
        self.layer_size_dict = layer_size_dict
        self.scenario = scenario
        self.cycle_num = cycle_num
        self.sigmoids = kwargs.get("sigmoids", None)
        kwargs.pop('sigmoids', None)

        assert hyper is not None
        self.threshold = hyper
        self.final_std = 1
        self.min_var = min_var
        self.num_class = num_class
        self.use = "AL"
        self.stat_statistics = kwargs.get("stat_statistics", True)
        self.weight = (0, 0)
        self.std = (1, 1)
        self.current = 0
        self.optimal_threshold = None
        # Joda Specific
        self.sep_metric = kwargs.get("sep_metric", "AB") # OOD filtering strategy
        self.seperator = kwargs.get("seperator", "metric") # for OOD filtering metric, kmeans, knn, knn-kmeans
        self.apply_ood_filter = kwargs.get("apply_ood_filter", True)
        self.use_balancing = kwargs.get("use_balancing", True) # class balancing

        self.coverage_set = set()
        self.mask_index_dict = {}
        self.use_nac = True
        if self.use_nac:
            self.min_var = 0

        # visualization
        self.plot_activation_histogram = False

        self.mean_dict = {}
        self.var_dict = {}
        self.layer_cache = {}
        self.shapley_dict = {}
        self.kde_cache = {}
        self.SA_cache = {}
        self.SA_history = torch.tensor([], device=self.device)
        self.layer_selection = None

        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(layer_size[0]).type(torch.LongTensor).to(self.device)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)

    def get_name(self):
        raise NotImplementedError

    def build(self, data_loader):
        """
        Builds the coverage model using the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader containing the training data.

        Returns:
            None
        """
        self.set_mask()
        # with torch.no_grad():
        gl_info('Building SA...')
        if self.seperator in ["kmeans", "knn", "knnkmeans"]:
            self.build_PCA(data_loader)
        else:
            for i, (data, label) in enumerate(tqdm(data_loader)):
                if isinstance(data, tuple):
                    data = (data[0].to(self.device), data[1].to(self.device))
                else:
                    data = data.to(self.device)
                label = label.to(self.device)
                self.build_database(data, label)
        self.to_numpy()

    def assess(self, data_loader, gain_mode="standard", surprise_strategy="surprise"):
        """
        Assess the coverage of the dataset using the given data loader.
        This will be done once on the labeled dataset to estimate teh

        Parameters:
        - data_loader (torch.utils.data.DataLoader): The data loader for the dataset.
        - gain_mode (str): The gain mode to use for assessment. Default is "standard".
        - surprise_strategy (str): The surprise strategy to use for assessment. Default is "surprise".

        Returns:
        - gain_list (torch.Tensor): The list of gains for in-distribution samples.
        - all_statistics (torch.Tensor): The statistics for all samples.
        """
        cached_SA_batches, cached_SA_labels, cached_SA_scores, cached_sample_idcs, cached_true_labels = self.build_greedy(
            data_loader, use_prediction=False, build=True)
        inD_indices = ((cached_true_labels >= 0) & (cached_true_labels < self.num_class)).nonzero(as_tuple=True)[0]
        # OOD_indices = (cached_SA_labels >= self.num_class).nonzero(as_tuple=True)[0]
        cached_SA_labels[inD_indices] = cached_true_labels[inD_indices]
        cached_dataset = torch.utils.data.TensorDataset(cached_SA_batches, cached_SA_labels, cached_SA_scores,
                                                        cached_sample_idcs, cached_true_labels)
        cached_dataloader = DataLoader(cached_dataset, batch_size=128, shuffle=False, num_workers=8)
        all_statistics = torch.tensor([])
        gain_list = torch.tensor([], device="cpu")
        for i, (SA_batch, pred_label_batch, score_batch, idcs_batch, true_labels_batch) in tqdm(
                enumerate(cached_dataloader)):
            if self.seperator in ["kmeans", "knn", "knnkmeans"]:
                SA_batch = self.scaler.transform(SA_batch)
                # SA_batch = self.pca.transform(SA_batch)
            else:
                SA_batch = SA_batch.numpy()
            dsa, statistics = self.calculate_greedy(SA_batch, pred_label_batch, score_batch,
                                                    surprise_strategy=surprise_strategy)
            # dsa, energy = self.step(data, label, surprise_strategy=surprise_strategy)
            if self.stat_statistics:
                ood_stats = (true_labels_batch < 0) * 2 + (true_labels_batch >= self.num_class)
                statistics = torch.cat((statistics, idcs_batch.unsqueeze(1), ood_stats.unsqueeze(1)), dim=1)
                all_statistics = torch.cat((all_statistics, statistics), dim=0)
            if dsa is None:
                continue
            gains = torch.cat([dsa.cpu().unsqueeze(1), statistics[:, 3].unsqueeze(1)], dim=1)
            gain_list = torch.cat([gain_list.cpu(), gains.cpu()], dim=0)
        mean_stats = torch.mean(gain_list, dim=0).float().tolist()
        mean_stats = tuple(mean_stats)
        std_stats = torch.std(gain_list, dim=0).float().tolist()
        std_stats = tuple(std_stats)

        if self.seperator == "metric" and "osal" in self.scenario and (self.cycle_num > 0 or self.scenario != "osal"):
            self.estimate_ood_threshold(all_statistics, mean_stats, std_stats)

        return gain_list[inD_indices], all_statistics

    def estimate_ood_threshold(self, all_statistics, mean_stats, std_stats):
        if self.sep_metric == "AB":
            metric = -(all_statistics[:, 0] + all_statistics[:, 1])
        elif self.sep_metric == "energy":
            metric = all_statistics[:, 3]
        elif self.sep_metric == "dsa":
            metric = all_statistics[:, 2]
        elif self.sep_metric == "dsaenergy":
            metric = min(1, mean_stats[0]) * (all_statistics[:, 3] - mean_stats[1]) / std_stats[1] + \
                     max(1.0 - mean_stats[0], 0.0) * (all_statistics[:, 2] - mean_stats[0]) / std_stats[0]
        else:
            raise NotImplementedError(f"metric: {self.sep_metric} is not supported")
        if self.scenario == "osal":
            ood_stats = (all_statistics[:, 6] == 1).float()
        else:
            ood_stats = (all_statistics[:, 6] == 2).float()
        fpr, tpr, thresholds = roc_curve(ood_stats, metric)
        youden_j = tpr - fpr
        optimal_threshold_index = youden_j.argmax()
        self.optimal_threshold = thresholds[optimal_threshold_index]
        gl_info(f"Optimal threshold: {self.optimal_threshold}")

    def top_k_loop(self, indices_and_gains: list, data_loader, coverage_al: str, no_label: bool = False,
                   gain_mode="standard", surprise_strategy="coverage", query_size=1000, cached_data=None):
        """
        Perform the top-k loop for active learning returning the topk most informative samples.

        Args:
            indices_and_gains (list): List of indices and their corresponding gains.
            data_loader: The data loader for iterating over the dataset.
            coverage_al (str): The coverage active learning strategy to use ("incremental" or "optimal").
            no_label (bool, optional): Whether to use labels during the active learning process. Defaults to False.
            gain_mode (str, optional): The gain mode to use ("standard" or "custom"). Defaults to "standard".
            surprise_strategy (str, optional): The surprise strategy to use ("coverage" or "energy"). Defaults to "coverage".
            query_size (int, optional): The number of samples to query in each iteration. Defaults to 1000.
            cached_data (None, optional): Cached data for optimization. Defaults to None.

        Returns:
            tuple or list: If coverage_al is "optimal", returns a tuple containing the selected sample's information.
                           If coverage_al is "incremental", returns a list of indices and gains, and all statistics.
        """

        # max_sample keeps the top sample by saving index, dsa value, predicted label and feature vector
        max_sample = (0, float("-inf"), 0, 0)
        # Loop through the data loader
        selected_indices = [idx for idx, _, _ in indices_and_gains]
        dataset_size = len(data_loader.dataset)
        all_statistics = torch.tensor([])
        for i, (SA_batch, pred_label_batch, score_batch, idcs_batch, true_labels) in tqdm(enumerate(data_loader)):
            # compute metric value (dsa) and statistics [dista, distb, dsa_raw, energy, gen]
            if self.seperator in ["kmeans", "knn", "knnkmeans"]:
                SA_batch = self.scaler.transform(SA_batch)
                # SA_batch = self.pca.transform(SA_batch)
            else:
                SA_batch = SA_batch.numpy()
            dsa, statistics = self.calculate_greedy(SA_batch, pred_label_batch, score_batch,
                                                    surprise_strategy=surprise_strategy)
            if coverage_al == "incremental":
                ood_stats = (true_labels < 0) * 2 + (true_labels >= self.num_class)
                statistics = torch.cat((statistics, idcs_batch.unsqueeze(1), ood_stats.unsqueeze(1)), dim=1)
                all_statistics = torch.cat((all_statistics, statistics), dim=0)
                # dsa, energy = self.step(sample_data, sample_target, gain_mode = gain_mode, surprise_strategy = surprise_strategy)
                indices_and_gains += torch.cat([idcs_batch.unsqueeze(1), dsa.unsqueeze(1)], dim=1).cpu().tolist()
            elif coverage_al == "optimal":
                # get the idx from idcs_batch where the max dsa is
                max_dsa, max_idx = torch.max(dsa, 0)
                if idcs_batch[max_idx] in selected_indices:
                    continue
                if max_dsa > max_sample[1]:
                    max_sample = (idcs_batch[max_idx], float(max_dsa), pred_label_batch[max_idx], SA_batch[max_idx])

        if coverage_al == "optimal":
            max_label = int(max_sample[3])
            self.SA_cache[max_label] = np.concatenate((self.SA_cache[max_label], max_sample[4].numpy()[None, :]),
                                                      axis=0)
            return max_sample

        return indices_and_gains, all_statistics

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        gl_info(f"feature_num:  {feature_num}")

    def build_PCA(self, data_loader):
        """
        build labeled data base and sets up kmeans, knn or knnkmeans seperator

        Args:
            data_loader: The data loader object.

        Returns:
            None
        """
        cached_SA_batches, _, cached_SA_scores, cached_sample_idcs, cached_SA_labels = self.build_greedy(data_loader,
                                                                                                         use_prediction=False,
                                                                                                         build=True)
        self.scaler = StandardScaler()
        gl_info("Building PCA and build data bank..")
        cached_SA_batches = self.scaler.fit_transform(cached_SA_batches)
        # self.pca = PCA(n_components=0.95)
        # cached_SA_batches = self.pca.fit_transform(cached_SA_batches)
        if "osal" in self.scenario and (self.cycle_num > 0 or self.scenario != "osal"):
            if self.scenario == "osal":
                inD_indices = ((cached_SA_labels >= 0) & (cached_SA_labels < self.num_class)).nonzero(as_tuple=True)[0]
                OOD_indices = (cached_SA_labels >= self.num_class).nonzero(as_tuple=True)[0]
            elif self.scenario == DataUpdateScenario.osal_near_far or self.scenario == DataUpdateScenario.osal_extending:
                inD_indices = ((cached_SA_labels >= 0)).nonzero(as_tuple=True)[0]
                OOD_indices = (cached_SA_labels < 0).nonzero(as_tuple=True)[0]
            if "kmeans" in self.seperator:
                init_centroids = np.array(
                    [cached_SA_batches[inD_indices].mean(0), cached_SA_batches[OOD_indices].mean(0)])
                self.kmeans = KMeans(n_clusters=2, init=init_centroids)
                self.kmeans.fit(cached_SA_batches)
                cluster_labels = self.kmeans.predict(cached_SA_batches)
                self.ind_cluster = int(np.argmax(np.bincount(cluster_labels[inD_indices])))
                self.ood_cluster = int(np.argmax(np.bincount(cluster_labels[OOD_indices])))
                if self.seperator == "knnkmeans":
                    if self.ind_cluster == self.ood_cluster:
                        self.seperator = "knn"
                    else:
                        self.seperator = "kmeans"
                gl_info(
                    f"Cluster labels: InD: {self.ind_cluster}, OOD: {self.ood_cluster}, Seperator: {self.seperator}")

        for i, label in enumerate(cached_SA_labels):
            if int(label.cpu()) >= 0 and int(label.cpu()) < self.num_class:
                if int(label.cpu()) in self.SA_cache.keys():
                    self.SA_cache[int(label.cpu())] += [cached_SA_batches[i]]
                else:
                    self.SA_cache[int(label.cpu())] = [cached_SA_batches[i]]
            elif (self.scenario == "osal" and int(label.cpu()) >= self.num_class) or \
                    (self.scenario == DataUpdateScenario.osal_near_far and int(label.cpu()) < 0) or \
                    (self.scenario == DataUpdateScenario.osal_extending and int(label.cpu()) < 0):
                if -1 in self.SA_cache.keys():
                    self.SA_cache[-1] += [cached_SA_batches[i]]
                else:
                    self.SA_cache[-1] = [cached_SA_batches[i]]

    def build_database(self, data_batch, label_batch):
        """
        build the labeled database if seperator is metric or scenario is not open-set
        (can in future be refactored to have only one function from SA_build and PCA_build)

        Args:
            data_batch (torch.Tensor): The input data batch.
            label_batch (torch.Tensor): The label batch.

        Returns:
            torch.Tensor: The SA batch.
        """
        SA_batch = []
        batch_size = label_batch.size(0)

        # Check if output of the model with Sigmoid activation is required or plain latent space.
        if self.use_nac:
            layer_output_dict, score = get_layer_output_nac(self.model, data_batch,
                                                                           layer_selection=self.layer_selection,
                                                                           use=self.use, dataset=self.dataset,
                                                                           sigmoids=self.sigmoids)
        else:
            layer_output_dict = get_layer_output(self.model, data_batch,
                                                                layer_selection=self.layer_selection)

        if self.plot_activation_histogram:
            for key, activations in layer_output_dict.items():
                if key not in self.layer_cache:
                    self.layer_cache[key] = activations.detach().cpu()
                else:
                    self.layer_cache[key] = torch.cat((self.layer_cache[key], activations.detach().cpu()), dim=0)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name and self.use_nac:
                continue
            else:
                SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1)  # [batch_size, num_neuron]
        # gl_info('SA_batch: ', SA_batch.size())
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        for i, label in enumerate(label_batch):
            if int(label.cpu()) >= 0 and int(label.cpu()) < self.num_class:
                if int(label.cpu()) in self.SA_cache.keys():
                    self.SA_cache[int(label.cpu())] += [SA_batch[i].detach().cpu().numpy()]
                else:
                    self.SA_cache[int(label.cpu())] = [SA_batch[i].detach().cpu().numpy()]
            elif (self.scenario == "osal" and int(label.cpu()) >= self.num_class) or \
                    (self.scenario == DataUpdateScenario.osal_near_far and int(label.cpu()) < 0) or \
                    (self.scenario == DataUpdateScenario.osal_extending and int(label.cpu()) < 0):
                if -1 in self.SA_cache.keys():
                    self.SA_cache[-1] += [SA_batch[i].detach().cpu().numpy()]
                else:
                    self.SA_cache[-1] = [SA_batch[i].detach().cpu().numpy()]

    def to_numpy(self):
        self.num_samples = 0
        for k in self.SA_cache.keys():
            self.SA_cache[k] = np.stack(self.SA_cache[k], 0)
            if k >= 0:
                self.num_samples += self.SA_cache[k].shape[0]

    def calculate(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(self.coverage_set)

    def coverage(self, cove_set):
        return len(cove_set)

    def compute_similarity(self, SA_batch, similarity_method="avg_similarity", beta=0.95):
        # Compute cosine similarity
        similarity = self.compute_cosine_similarity(SA_batch)

        # Calculate the desired similarity based on the chosen method
        if similarity_method == "avg_similarity" or similarity_method == "similarity":
            result = torch.mean(similarity, dim=1)
        elif similarity_method == "max_similarity":
            result = torch.max(similarity, dim=1).values
        elif similarity_method == "exp_similarity":
            result = self.compute_exp_weighted_average_similarity(similarity, beta)
        else:
            raise ValueError("Invalid similarity method.")

        return result

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def compute_cosine_similarity(self, SA_batch, SA_bank=None):
        # Reshape SA_batch to have dimensions (batch_size, 1, num_neurons)
        SA_batch = SA_batch.to(device=self.device)
        if SA_bank == None:
            SA_bank = self.SA_history
        if SA_bank.numel() == 0:  # Check if SA_history is empty
            return torch.zeros(SA_batch.size(0), SA_batch.size(0),
                               device=self.device)  # Return zeros for similarity if SA_history is empty

        # Compute cosine similarity using PyTorch's cosine_similarity function
        similarity = self.sim_matrix(SA_batch, SA_bank.to(self.device))

        return similarity.cpu()

    def compute_exp_weighted_average_similarity(self, similarity, beta):
        # Calculate the exponentially weighted average similarity for each sample in SA_batch
        num_samples = self.SA_history.size(0)
        exp_weights = torch.tensor([beta ** (num_samples - i - 1) for i in range(num_samples)],
                                   device=self.SA_history.device)
        exp_weights = exp_weights / exp_weights.sum()

        exp_weighted_avg_similarity = torch.matmul(similarity, exp_weights)

        return exp_weighted_avg_similarity

    def update_SA_history(self, SA_batch):
        # Concatenate SA_batch with SA_history along the 0th dimension
        self.SA_history = torch.cat([self.SA_history, SA_batch], dim=0)

    def top_k(self, data_loader, k, coverage_al, no_label=False, gain_mode="standard", surprise_strategy="coverage"):
        """
        Selects the top k samples based on a coverage-based active learning strategy

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader for the dataset.
            k (int): The number of samples to select.
            coverage_al (str): The coverage-based active learning strategy to use ("optimal" or "incremental").
            no_label (bool, optional): Whether to consider samples without labels. Defaults to False.
            gain_mode (str, optional): The gain mode to use ("standard" or "custom"). Defaults to "standard".
            surprise_strategy (str, optional): The surprise strategy to use ("coverage" or "random"). Defaults to "coverage".

        Returns:
            tuple: A tuple containing the top gains, top indices, and statistics (if applicable).
        """
        # Initialize a min-heap to keep track of the top k indices with highest gains
        indices_and_gains = []
        #top_indices_heap = []
        cached_SA_batches, cached_SA_labels, cached_SA_scores, cached_sample_idcs, cached_true_labels = self.build_greedy(data_loader, use_prediction=True)
        cached_dataset = torch.utils.data.TensorDataset(cached_SA_batches, cached_SA_labels, cached_SA_scores, cached_sample_idcs, cached_true_labels)
        cached_dataloader = DataLoader(cached_dataset, batch_size=128, shuffle=False, num_workers=8)
        if coverage_al == "optimal":
            for _ in tqdm(range(k)):
                top_sample, _ = self.top_k_loop(indices_and_gains, cached_dataloader, coverage_al, query_size=k)[:3]
                # self.SA_cache[top_sample.label].append(top_sample.data)
                indices_and_gains.append(top_sample)
        elif coverage_al == "incremental":
            indices_and_gains, statistics = self.top_k_loop(indices_and_gains, cached_dataloader, coverage_al, no_label=no_label, gain_mode=gain_mode, surprise_strategy=surprise_strategy, query_size=k)

        if "osal" in self.scenario:
            p, r = self.calculate_ood_accuracy(indices_and_gains, cached_true_labels, self.scenario)
            gl_info(f"OOD Detection Precision: {p}, Recall: {r}")
        indices_and_gains.sort(key=lambda x: x[1], reverse=True)
        top_indices = [int(idx) for idx, _ in indices_and_gains]
        top_gains = [gain.cpu().item() if torch.is_tensor(gain) else gain for _, gain in indices_and_gains]
        return top_gains, top_indices, statistics


def get_layer_output_nac(model, data, layer_selection = None, use="AL", dataset="cifar10", sigmoids = None):
    name_counter = {}
    layer_output_dict = {}
    layer_dict = get_model_layers(model, mode="nac")

    def hook(module, input, output):
        class_name = module.__class__.__name__
        if class_name not in name_counter.keys():
            name_counter[class_name] = 1
        else:
            name_counter[class_name] += 1
        layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output

    hooks = []
    for layer, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    try:
        if use == "AL":
            final_out, _, _ = model(data)
        else:
            final_out = model(data)
    finally:
        for h in hooks:
            h.remove()
    layer_selection = ["AdaptiveAvgPool2d-1", "Sequential-3", "Sequential-2", "Sequential-1", "Linear-1"]
    if layer_selection is not None:
        layer_output_dict = filter_dict_by_keys(layer_selection, layer_output_dict)
    unrolled_layer_output_dict = {}
    for k in layer_output_dict.keys():
        unrolled_layer_output_dict[k] = layer_output_dict[k]
    if dataset=="cifar10":
        sig_alpha = {"AdaptiveAvgPool2d-1": 100, "Sequential-3": 1000, "Sequential-2": 0.001, "Sequential-1": 0.001}
    elif dataset=="cifar100":
        #sig_alpha = {"AdaptiveAvgPool2d-1": 1, "Sequential-3": 0.1, "Sequential-2": 0.1, "Sequential-1": 0.1}
        sig_alpha = {"AdaptiveAvgPool2d-1": 50, "Sequential-3": 10, "Sequential-2": 1, "Sequential-1": 0.005}
    elif dataset=="imagenet":
        sig_alpha = {"AdaptiveAvgPool2d-1": 3000, "Sequential-3": 300, "Sequential-2": 0.01, "Sequential-1": 1}
    if sigmoids:
        sig_alpha = sigmoids

    for layer, output in unrolled_layer_output_dict.items():
        if len(output.size()) == 4: # (N, K, H, w)
            retain_graph = False if "layer1" in layer else True
            output_kl_grad = kl_grad(output, final_out, retain_graph = retain_graph)
            output = output.mean((2, 3))
            output_kl_grad = output_kl_grad.mean((2,3))
            #if use == "AL":
            #    output = output * output_kl_grad
            #else:
            output = sigmoid(output * output_kl_grad, sig_alpha = sig_alpha[layer])
        unrolled_layer_output_dict[layer] = output.detach()
    return unrolled_layer_output_dict, final_out


def get_model_layers(model, mode = "cov"):
    layer_dict = {}
    name_counter = {}
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [], mode)
        assert len(name_list) == len(module_list)
        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            layer_dict['%s-%d' % (class_name, name_counter[class_name])] = module
    # DEBUG
    # print('layer name')
    # for k in layer_dict.keys():
    #     print(k, ': ', layer_dict[k])
    return layer_dict


def iterate_module(name, module, name_list, module_list, mode = "cov"):
    if is_valid(module, mode):
        return name_list + [name], module_list + [module]
    else:

        if len(list(module.named_children())):
            for child_name, child_module in module.named_children():
                name_list, module_list = \
                    iterate_module(child_name, child_module, name_list, module_list)
        return name_list, module_list

def is_valid(module, mode = "cov"):
    if mode == "cov":
        return (isinstance(module, nn.Linear)
                or isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Conv1d)
                or isinstance(module, nn.Conv3d)
                or isinstance(module, nn.RNN)
                or isinstance(module, nn.LSTM)
                or isinstance(module, nn.GRU)
                )
    else:
        return (isinstance(module, nn.Sequential)
                or isinstance(module, nn.AdaptiveAvgPool2d)
                or isinstance(module, nn.Linear)
                )


def filter_dict_by_keys(input_list, input_dict):
    sub_dict = {key: input_dict[key] for key in input_list if key in input_dict}
    return sub_dict


def kl_grad(b_state, outputs, temperature=1.0, retain_graph=False, **kwargs):
    """
    This implementation follows https://github.com/deeplearning-wisc/gradnorm_ood
    """
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    num_classes = outputs.shape[-1]
    targets = torch.ones_like(outputs) / num_classes

    loss = (torch.sum(-targets * logsoftmax(outputs), dim=-1))
    layer_grad = torch.autograd.grad(loss.sum(), b_state, create_graph=False,
                                     retain_graph=retain_graph, **kwargs)[0]
    return layer_grad


def sigmoid(x, sig_alpha=1.0):
    """
    sig_alpha is the steepness controller (larger denotes steeper)
    """
    return 1 / (1 + torch.exp(-sig_alpha * x))


def get_layer_output(model, data, pad_length=None, layer_selection = None):
    with torch.no_grad():
        name_counter = {}
        layer_output_dict = {}
        layer_dict = get_model_layers(model)

        def hook(module, input, output):
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
                layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output[0]
            else:
                layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            final_out = model(data)
        finally:
            for h in hooks:
                h.remove()
        if layer_selection is not None:
            layer_output_dict = filter_dict_by_keys(layer_selection, layer_output_dict)
        unrolled_layer_output_dict = {}
        for k in layer_output_dict.keys():
            if ('RNN' in k) or ('LSTM' in k) or ('GRU' in k):
                assert pad_length == len(layer_output_dict[k])
                for i in range(pad_length):
                    unrolled_layer_output_dict['%s-%d' % (k, i)] = layer_output_dict[k][i]
            else:
                unrolled_layer_output_dict[k] = layer_output_dict[k]

        for layer, output in unrolled_layer_output_dict.items():
            if len(output.size()) == 4: # (N, K, H, w)
                output = output.mean((2, 3))
            unrolled_layer_output_dict[layer] = output.detach()
        return unrolled_layer_output_dict


class SisomLogic():
    def init_variable(self, hyper, min_var=1e-5, num_class=10):
        self.name = self.get_name()
        assert self.name in ['LSC', 'DSC', 'MDSC']
        assert hyper is not None
        self.threshold = hyper
        self.min_var = min_var
        self.num_class = num_class
        self.use = "AL"
        self.data_count = 0
        self.weight = (0, 0)
        self.std = (1, 1)
        self.current = 0
        self.coverage_set = set()
        self.mask_index_dict = {}
        self.use_shapley = False
        self.use_nac = True
        if self.use_nac:
            self.min_var = 0
        self.plot_activation_histogram = False
        self.mean_dict = {}
        self.var_dict = {}
        self.layer_cache = {}
        self.shapley_dict = {}
        self.kde_cache = {}
        self.SA_cache = {}
        self.SA_history = torch.tensor([], device=self.device)
        self.layer_selection = None
        # self.layer_selection = [
        #                         #"Conv2d-1", "Conv2d-2", "Conv2d-3", "Conv2d-4", "Conv2d-5",
        #                         #"Conv2d-6", "Conv2d-7", "Conv2d-8", "Conv2d-9", "Conv2d-10",
        #                         #"Conv2d-11", "Conv2d-12", "Conv2d-13", "Conv2d-14", "Conv2d-15",
        #                         "Conv2d-16", "Conv2d-17", "Conv2d-18", "Conv2d-19", "Conv2d-20",
        #                         "Linear-1"
        #                         ]
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(layer_size[0]).type(torch.LongTensor).to(self.device)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
        # for (layer_name, layer_size) in self.layer_size_dict.items():
        #     if "Conv" in layer_name:
        #         self.mask_index_dict[layer_name] = torch.ones(layer_size[0]*4).type(torch.LongTensor).to(self.device)
        #         self.mean_dict[layer_name] = torch.zeros(layer_size[0]*4).to(self.device)
        #         self.var_dict[layer_name] = torch.zeros(layer_size[0]*4).to(self.device)
        #     else:
        #         self.mask_index_dict[layer_name] = torch.ones(layer_size[0]).type(torch.LongTensor).to(self.device)
        #         self.mean_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
        #         self.var_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)

    def get_name(self):
        raise NotImplementedError

    def build(self, data_loader):
        if self.use_shapley:
            label_count = {}
            print("Building shapley values...")
            for i, (data, label) in enumerate(tqdm(data_loader)):
                # print(data.size())
                if int(label) in label_count.keys():
                    label_count[int(label)] += 1
                else:
                    label_count[int(label)] = 1
                if isinstance(data, tuple):
                    data = (data[0].to(self.device), data[1].to(self.device))
                else:
                    data = data.to(self.device)
                self.set_shapley_values(data, label)
            for (label, layer_output) in self.shapley_dict.items():
                self.shapley_dict[label] = (layer_output / label_count[label])
        else:
            # with torch.no_grad():
            print('Building Mean & Var...')
            if self.min_var > 0:
                for i, (data, label) in enumerate(tqdm(data_loader)):
                    # print(data.size())
                    if isinstance(data, tuple):
                        data = (data[0].to(self.device), data[1].to(self.device))
                    else:
                        data = data.to(self.device)
                    self.set_meam_var(data, label)
            self.set_mask()
        # with torch.no_grad():
        print('Building SA...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.build_SA(data, label)
        self.to_numpy()
        if self.plot_activation_histogram:
            self.track_all_activations(self.layer_cache, self.cycle_num)
            self.layer_cache = {}
        if self.name == 'LSC':
            self.set_kde()
        if self.name == 'MDSC':
            self.to_numpy()
            res = self.estimator.class_wise_avg_cov(self.SA_cache)
            self.estimator.update(res)
            self.compute_covinv()

    def assess(self, data_loader, gain_mode="standard", surprise_strategy="surprise"):
        if ("energy" in gain_mode):
            surprise_strategy = "surprise"
        gain_list = torch.tensor([], device="cpu")
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            dsa, energy = self.step(data, label, surprise_strategy=surprise_strategy)
            gains = torch.cat([dsa.cpu().unsqueeze(1), energy.cpu().unsqueeze(1)], dim=1)
            gain_list = torch.cat([gain_list.cpu(), gains.cpu()], dim=0)

        return gain_list

    def step(self, data, label, gain_mode="standard", surprise_strategy="coverage", coverage_al="incremental"):
        dsa, energy, layer_output_dict = self.calculate(data, label, surprise_strategy)
        # raw_gain, sample_gain = self.gain(cove_set, layer_output_dict, gain_mode, surprise_strategy)
        return dsa, energy

    def set_shapley_values(self, data, label):
        shap_list = []
        batch_size = label.size(0)
        assert batch_size == 1, "Backward Call can only be made one by one for each sample"
        fw_dict, bw_dict = get_layer_fw_bw_output(self.model, data, label, layer_selection=self.layer_selection)
        for (layer_name, layer_output) in fw_dict.items():
            #     if "Linear" not in layer_name:
            #         shap_list.append(torch.abs(torch.mul(layer_output, bw_dict[layer_name])))
            shap_list.append(torch.abs(torch.mul(layer_output, bw_dict[layer_name])))

        shap_list = torch.cat(shap_list, 1).detach().cpu()  # [batch_size, num_neuron]

        if int(label.item()) in self.shapley_dict.keys():
            self.shapley_dict[int(label.item())] += shap_list
        else:
            self.shapley_dict[int(label.item())] = shap_list

    def top_k_loop(self, indices_and_gains: list, data_loader, coverage_al: str, no_label: bool = False,
                   gain_mode="standard", surprise_strategy="coverage", query_size=1000, cached_data=None):
        # max_sample keeps the top sample by saving gain, index and data
        max_sample = (-1, -1.0, -1)
        # Loop through the data loader
        selected_indices = [idx for idx, _, _ in indices_and_gains]
        dataset_size = len(data_loader.dataset)
        for i in tqdm(range(cached_data[0].shape[0])):
            # Get the index for the current sample
            sample_idx = i
            if coverage_al == "incremental":
                dsa, energy = self.calculate_greedy(cached_data[0][sample_idx][None, :],
                                                    cached_data[1][sample_idx].unsqueeze(0),
                                                    cached_data[2][sample_idx].unsqueeze(0),
                                                    surprise_strategy=surprise_strategy)
                # dsa, energy = self.step(sample_data, sample_target, gain_mode = gain_mode, surprise_strategy = surprise_strategy)
                indices_and_gains.append((sample_idx, dsa, energy[0]))
            elif coverage_al == "optimal":
                if sample_idx in selected_indices:
                    continue
                dsa, energy = self.calculate_greedy(cached_data[0][sample_idx][None, :],
                                                    cached_data[1][sample_idx].unsqueeze(0),
                                                    cached_data[2][sample_idx].unsqueeze(0),
                                                    surprise_strategy=surprise_strategy)
                if dsa > max_sample[1]:
                    max_sample = (sample_idx, float(dsa), float(energy))
        if coverage_al == "optimal":
            max_label = int(cached_data[1][max_sample[0]])
            self.SA_cache[max_label] = np.concatenate(
                (self.SA_cache[max_label], cached_data[0][max_sample[0]][None, :]), axis=0)
            return max_sample

        return indices_and_gains

    def top_k_oneshot(self, indices_and_gains: list, data_loader, coverage_al: str, no_label: bool = False,
                   gain_mode="standard", surprise_strategy="coverage", query_size=1000, cached_data=None):
        """
        :param indices_and_gains:
        :param data_loader:
        :param coverage_al:
        :param no_label:
        :param gain_mode:
        :param surprise_strategy:
        :param query_size:
        :param cached_data:

        :return indices_and_gains
        """
        SA_batch, label_batch, score = cached_data[0], cached_data[1].to(int),cached_data[2]
        SA_batch = torch.from_numpy(SA_batch)
        indicies = list(range(label_batch.shape[0]))
        for _ in range(query_size):
            dsa2, energy2 = self.calculate_greedy_full(SA_batch,
                                                       label_batch,
                                                       score,
                                                       surprise_strategy=surprise_strategy)
            max_idx=np.argmax(dsa2)
            das_max=np.max(dsa2)
            en_max=np.max(energy2)
            max_sample = (indicies[max_idx], float(das_max), en_max)
            del indicies[max_idx]
            indices_and_gains.append(max_sample)
            self.SA_cache[int(label_batch[max_idx])] = np.concatenate(
                (self.SA_cache[int(label_batch[max_idx])], SA_batch[max_idx][None, :]), axis=0)
            label_batch = torch.cat((label_batch[:max_idx], label_batch[max_idx + 1:]))
            SA_batch = torch.cat((SA_batch[:max_idx], SA_batch[max_idx + 1:]))
            score = torch.cat((score[:max_idx], score[max_idx + 1:]))

        return indices_and_gains


    def top_k(self, data_loader, k, coverage_al, no_label=False, gain_mode="standard", surprise_strategy="coverage"):
        # Initialize a min-heap to keep track of the top k indices with highest gains
        indices_and_gains = []
        # top_indices_heap = []
        cached_SA_batches, cached_SA_labels, cached_SA_scores = self.build_greedy(data_loader)
        cached_data = (cached_SA_batches, cached_SA_labels, cached_SA_scores)
        if coverage_al == "optimal":
            for _ in tqdm(range(k)):
                top_sample = self.top_k_loop(indices_and_gains, data_loader, coverage_al, query_size=k,
                                             cached_data=cached_data)
                # self.SA_cache[top_sample.label].append(top_sample.data)
                indices_and_gains.append(top_sample)
        elif coverage_al == "optimal2":
            indices_and_gains = self.top_k_oneshot(indices_and_gains, data_loader, coverage_al, query_size=k,
                                         cached_data=cached_data)
            # self.SA_cache[top_sample.label].append(top_sample.data)
        elif coverage_al == "individual" or coverage_al == "incremental":
            indices_and_gains = self.top_k_loop(indices_and_gains, data_loader, coverage_al, no_label=no_label,
                                                gain_mode=gain_mode, surprise_strategy=surprise_strategy, query_size=k,
                                                cached_data=cached_data)

        # Sort the top k indices by their order of appearance in the data loader
        # top_indices_and_gains = sorted([(idx, gain) for gain, idx in top_indices_heap])

        indices_and_gains.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _, _ in indices_and_gains]
        top_gains = [gain.cpu().item() if torch.is_tensor(gain) else gain for _, gain, _ in indices_and_gains]
        top_energies = [energy.cpu().item() if torch.is_tensor(energy) else energy for _, _, energy in
                        indices_and_gains]
        return top_gains, top_indices

    def build_greedy(self, data_loader):
        # cached SA_batches contains all SA_batches, has size (num_samples, num_neurons)
        # initialize SA_batches as empty numpy array
        cached_SA_batches = torch.tensor([])
        cached_SA_labels = torch.tensor([])
        cached_SA_scores = torch.tensor([])
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            SA_batch, label_batch, layer_output_dict, score = self.nac_forward_pass(data, label,
                                                                                    surprise_strategy="surprise",
                                                                                    use_prediction=True)
            # SA_batch is a numpy array of shape (batch_size, num_neurons)
            cached_SA_batches = torch.cat((cached_SA_batches, SA_batch), dim=0)
            cached_SA_labels = torch.cat((cached_SA_labels, label_batch), dim=0)
            cached_SA_scores = torch.cat((cached_SA_scores, score), dim=0)
        return cached_SA_batches.numpy(), cached_SA_labels, cached_SA_scores

    def set_meam_var(self, data, label):
        batch_size = label.size(0)
        if self.use_nac:
            layer_output_dict, score = get_layer_output_nac(self.model, data, layer_selection=self.layer_selection,
                                                                 use=self.use, dataset=self.dataset,
                                                                 sigmoids=self.sigmoids)
        else:
            layer_output_dict = get_layer_output(self.model, data, layer_selection=self.layer_selection)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name and self.use_nac:
                continue
            else:
                self.data_count += batch_size
                self.mean_dict[layer_name] = ((self.data_count - batch_size) * self.mean_dict[
                    layer_name] + layer_output.sum(0)) / self.data_count
                self.var_dict[layer_name] = (self.data_count - batch_size) * self.var_dict[layer_name] / self.data_count \
                                            + (self.data_count - batch_size) * (
                                                        (layer_output - self.mean_dict[layer_name]) ** 2).sum(
                    0) / self.data_count ** 2

    def get_name(self):
        return 'DSC'

    def nac_forward_pass(self, data_batch, label_batch, surprise_strategy="surprise", use_prediction=False):
        SA_batch = []
        if self.use_nac:
            layer_output_dict, score = get_layer_output_nac(self.model, data_batch,
                                                                 layer_selection=self.layer_selection, use=self.use,
                                                                 dataset=self.dataset, sigmoids=self.sigmoids)
        else:
            layer_output_dict = get_layer_output(self.model, data_batch, layer_selection=self.layer_selection)

        # it has to be considered that the actual label is supposed to be unknown during inference
        # thus, one should use the predicted label for computing the SA-Coverage instead
        # TODO: generalize later, for now, assume classification layer is called "linear-1"
        if label_batch is not None:
            batch_size = label_batch.size(0)
        elif not self.use_nac:
            label_batch = torch.tensor(layer_output_dict["Linear-1"].argmax().item(), device=self.device).unsqueeze(0)
            batch_size = 1
        else:
            label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
            batch_size = 1
        if use_prediction:
            if not self.use_nac:
                label_batch = torch.tensor(layer_output_dict["Linear-1"].argmax(dim=1).tolist(), device=self.device)
                if batch_size == 1:
                    label_batch = label_batch.unsqueeze(0)
            else:
                label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
                batch_size = len(label_batch)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name and self.use_nac:
                continue
            else:
                SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach().cpu()  # [batch_size, num_neuron]

        return SA_batch, label_batch.detach().cpu(), layer_output_dict, score.detach().cpu()

    def calculate(self, data_batch, label_batch, surprise_strategy="coverage", use_prediction=False):
        cove_set = set()
        dsa_list = []
        energy_list = []
        SA_batch = []
        if self.use_nac:
            layer_output_dict, score = get_layer_output_nac(self.model, data_batch,
                                                                 layer_selection=self.layer_selection, use=self.use,
                                                                 dataset=self.dataset, sigmoids=self.sigmoids)
        else:
            layer_output_dict = get_layer_output(self.model, data_batch, layer_selection=self.layer_selection)

        # it has to be considered that the actual label is supposed to be unknown during inference
        # thus, one should use the predicted label for computing the SA-Coverage instead
        # TODO: generalize later, for now, assume classification layer is called "linear-1"
        if label_batch is not None:
            batch_size = label_batch.size(0)
        elif not self.use_nac:
            label_batch = torch.tensor(layer_output_dict["Linear-1"].argmax().item(), device=self.device).unsqueeze(0)
            batch_size = 1
        else:
            label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
            batch_size = 1
        if use_prediction:
            if not self.use_nac:
                label_batch = torch.tensor(layer_output_dict["Linear-1"].argmax(dim=1).tolist(), device=self.device)
                if batch_size == 1:
                    label_batch = label_batch.unsqueeze(0)
            else:
                label_batch = torch.tensor(score.argmax(dim=1).tolist(), device=self.device)
                batch_size = len(label_batch)

        for (layer_name, layer_output) in layer_output_dict.items():
            if "Linear" in layer_name and self.use_nac:
                continue
            else:
                SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy()  # [batch_size, num_neuron]
        energy = get_energy_score(score).cpu().detach().numpy()
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]

            # # using numpy
            # dist_a_list = np.linalg.norm(SA - self.SA_cache[int(label.cpu())], axis=1)
            # idx_a = np.argmin(dist_a_list, 0)
            if self.use_shapley:
                dist_a_list = torch.linalg.norm(
                    self.shapley_dict[int(label.cpu())].to(self.device) *
                    (torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                        self.device)),
                    dim=1
                )
            else:
                dist_a_list = torch.linalg.norm(
                    torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                        self.device), dim=1)
            if self.weight[1] == 0:
                idx_a = torch.topk(dist_a_list, 2, dim=0, largest=False)[1][1]
            else:
                idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    # # using numpy
                    # dist_b_list += np.linalg.norm(SA - self.SA_cache[j], axis=1).tolist()
                    if self.use_shapley:
                        dist_b_list += torch.linalg.norm(
                            self.shapley_dict[j].to(self.device) * (
                                        torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(
                                    self.device)),
                            dim=1).cpu().numpy().tolist()
                    else:
                        dist_b_list += torch.linalg.norm(
                            torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                            dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            dsa = min(self.weight[0], 1.0) * ((energy[i] - self.weight[1]) / self.std[1]) + \
                  max(1.0 - self.weight[0], 0.0) * ((dsa - self.weight[0]) / self.std[1])
            # dsa = energy[i]
            if surprise_strategy == "dist_a":
                dsa_list.append(float(dist_a))
            elif surprise_strategy == "dist_b":
                if dist_b > 0:
                    db = 1 / dist_b
                else:
                    db = 1e-6
                dsa_list.append(float(db))
            else:
                dsa_list.append(dsa)
                energy_list.append(energy[i])
            if (not np.isnan(dsa)) and (not np.isinf(dsa)):
                cove_set.add(int(dsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        if surprise_strategy == "coverage":
            return cove_set, layer_output_dict
        else:
            if batch_size == 1:
                if surprise_strategy == "dist_a":
                    return dist_a, layer_output_dict
                elif surprise_strategy == "dist_b":
                    return db, layer_output_dict
                else:
                    return dsa, energy, layer_output_dict
            else:
                return torch.tensor(dsa_list).to(self.device), torch.tensor(energy_list).to(
                    self.device), layer_output_dict

    def calculate_greedy(self, SA_batch, label_batch, score, surprise_strategy="surprise"):
        cove_set = set()
        dsa_list = []
        batch_size = label_batch.size(0)
        energy_list = []
        energy = get_energy_score(score).cpu().detach().numpy()
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]

            # # using numpy
            # dist_a_list = np.linalg.norm(SA - self.SA_cache[int(label.cpu())], axis=1)
            # idx_a = np.argmin(dist_a_list, 0)
            if self.use_shapley:
                dist_a_list = torch.linalg.norm(
                    self.shapley_dict[int(label.cpu())].to(self.device) *
                    (torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                        self.device)),
                    dim=1
                )
            else:
                dist_a_list = torch.linalg.norm(
                    torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(
                        self.device), dim=1)
            if self.weight[1] == 0:
                idx_a = torch.topk(dist_a_list, 2, dim=0, largest=False)[1][1]
            else:
                idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if (j != int(label.cpu())) and (j in self.SA_cache.keys()):
                    # # using numpy
                    # dist_b_list += np.linalg.norm(SA - self.SA_cache[j], axis=1).tolist()
                    if self.use_shapley:
                        dist_b_list += torch.linalg.norm(
                            self.shapley_dict[j].to(self.device) * (
                                        torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(
                                    self.device)),
                            dim=1).cpu().numpy().tolist()
                    else:
                        dist_b_list += torch.linalg.norm(
                            torch.from_numpy(SA_a).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                            dim=1).cpu().numpy().tolist()

            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            dsa = min(self.weight[0], 1.0) * ((energy[i] - self.weight[1]) / self.std[1]) + \
                  max(1.0 - self.weight[0], 0.0) * ((dsa - self.weight[0]) / self.std[1])
            # dsa = energy[i]
            if surprise_strategy == "dist_a":
                dsa_list.append(float(dist_a))
            elif surprise_strategy == "dist_b":
                if dist_b > 0:
                    db = 1 / dist_b
                else:
                    db = 1e-6
                dsa_list.append(float(db))
            else:
                dsa_list.append(dsa)
                energy_list.append(energy[i])
            if (not np.isnan(dsa)) and (not np.isinf(dsa)):
                cove_set.add(int(dsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        if batch_size == 1:
            if surprise_strategy == "dist_a":
                return dist_a
            elif surprise_strategy == "dist_b":
                return db
            else:
                return dsa, energy
        else:
            return torch.tensor(dsa_list).to(self.device), torch.tensor(energy_list).to(self.device)

def get_energy_score(logits):
    return -torch.logsumexp(logits, dim=1)

def get_layer_fw_bw_output(model, data, label, layer_selection = None):
    name_counter = {}
    layer_output_dict_fw = {}
    layer_output_dict_bw = {}
    layer_dict = get_model_layers(model)

    def fw_hook(module, input, output):
        class_name = module.__class__.__name__
        if class_name not in name_counter.keys():
            name_counter[class_name] = 1
        else:
            name_counter[class_name] += 1
        if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
            layer_output_dict_fw['%s-%d' % (class_name, name_counter[class_name])] = output[0]
        else:
            layer_output_dict_fw['%s-%d' % (class_name, name_counter[class_name])] = output

    def bw_hook(module, grad_input, grad_output):
        class_name = module.__class__.__name__
        layer_output_dict_bw['%s-%d' % (class_name, name_counter[class_name])] = grad_output[0]
        name_counter[class_name] -= 1

    hooks = []
    for layer, module in layer_dict.items():
        hooks.append(module.register_forward_hook(fw_hook))
        hooks.append(module.register_full_backward_hook(bw_hook))
    try:
        model.zero_grad()
        final_out = model(data)
        #final_out, _, _ = model(data)
        #name_counter = {}
        final_out[0, label].backward(retain_graph=True)
    finally:
        for h in hooks:
            h.remove()

    if layer_selection is not None:
        layer_output_dict_fw = filter_dict_by_keys(layer_selection, layer_output_dict_fw)
        layer_output_dict_bw = filter_dict_by_keys(layer_selection, layer_output_dict_bw)
    for layer, output in layer_output_dict_fw.items():
        bw_output = layer_output_dict_bw[layer]
        if len(output.size()) == 4: # (N, K, H, w)
            output = output.mean((2, 3))
            bw_output = bw_output.mean((2,3))
            # _, K, H, W = output.size()
            # if (K, H, W) == (256, 8, 8):
            #     kernel_size = (4, 4)
            #     stride = (4, 4)
            # elif (K, H, W) == (512, 4, 4):
            #     kernel_size = (2, 2)
            #     stride = (2, 2)
            # else:
            #     raise ValueError("Unsupported tensor shape.")

            # # Apply maxpooling
            # if "Conv" in layer:
            #     output = F.max_pool2d(output, kernel_size=kernel_size, stride=stride)
            # output = output.view(output.size(0), -1)
        layer_output_dict_fw[layer] = output.detach()
        layer_output_dict_bw[layer] = bw_output.detach()

    return layer_output_dict_fw, layer_output_dict_bw

def get_layer_output_sizes(model, data, pad_length=None, mode="cov"):
    output_sizes = {}
    hooks = []
    name_counter = {}
    if mode=="cov":
        layer_dict = get_model_layers(model)
    else:
        layer_dict = get_model_layers(model, mode="nac")
    def hook(module, input, output):
        class_name = module.__class__.__name__
        if class_name not in name_counter.keys():
            name_counter[class_name] = 1
        else:
            name_counter[class_name] += 1
        if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
            output_sizes['%s-%d' % (class_name, name_counter[class_name])] = [output[0].size(2)]
        else:
            output_sizes['%s-%d' % (class_name, name_counter[class_name])] = list(output.size()[1:])

    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    try:
        model(data)
    finally:
        for h in hooks:
            h.remove()

    unrolled_output_sizes = {}
    for k in output_sizes.keys():
        if ('RNN' in k) or ('LSTM' in k) or ('GRU' in k):
            for i in range(pad_length):
                unrolled_output_sizes['%s-%d' % (k, i)] = output_sizes[k]
        else:
            unrolled_output_sizes[k] = output_sizes[k]
    # DEBUG
    # print('output size')
    # for k in output_sizes.keys():
    #     print(k, ': ', output_sizes[k])
    return unrolled_output_sizes