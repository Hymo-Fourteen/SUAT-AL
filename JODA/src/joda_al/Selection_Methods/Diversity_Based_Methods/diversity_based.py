import itertools
from typing import Dict, Union, Tuple, List, Optional
from warnings import warn

# import deprecation
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
from torch import nn

from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.data_loaders.sampler import SubsetSequentialSampler
from joda_al.Selection_Methods.Diversity_Based_Methods.kcenterGreedy import kCenterGreedy
from joda_al.models.model_lib import ModelHandler
from joda_al.Selection_Methods.query_method_def import QueryMethod

from joda_al.task_supports.pytorch.TaskHandler import TaskHandler


def select_batch_coreset(features, selected_samples, query_size, gpu=False, break_on_overlap=False):
    if isinstance(selected_samples, list):
        selected_size = len(selected_samples)
    else:
        selected_size = selected_samples.shape[0]

    def update_distances(cluster_centers, min_distances):
        x = features[cluster_centers]
        # Update min_distances for all examples given new cluster center.
        dist = pairwise_distances(features, x, metric='euclidean')
        if min_distances is None:
            min_distances = np.min(dist, axis=1).reshape(-1, 1)
        else:
            min_distances = np.minimum(min_distances, dist)
        return min_distances

    def update_distances_torch(cluster_centers, min_distances):
        feature = torch.tensor(features)
        cluster_centers = [int(c) for c in cluster_centers]
        x = feature[cluster_centers]
        # Update min_distances for all examples given new cluster center.

        dist = pairwise_distances_torch(feature, x)
        # print(dist.shape,feature.shape,x.shape)
        if min_distances is None:
            min_distances, min_idx = torch.min(dist, dim=1)
            min_distances = min_distances.reshape(-1, 1)
        else:
            min_distances = torch.minimum(min_distances, dist)
        return min_distances

    min_distances = None
    if gpu:
        f_argmax = torch.argmax
        features = torch.tensor(features).cuda()
        f_dist = update_distances_torch
    else:
        f_argmax = np.argmax
        f_dist = update_distances

    # init clusters
    min_distances = f_dist(selected_samples, min_distances)
    print(min_distances)
    new_batch = []
    new_batch_distance = []
    if query_size > (features.shape[0] - selected_size):
        print('Requested Batch Size large than available data')
        query_size = (features.shape[0] - selected_size)
    for _ in range(query_size):
        ind = f_argmax(min_distances)
        # New examples should not be in already selected since those points
        # should have min_distance of zero to a cluster center.
        # assert ind not in selected_samples
        print(ind)
        if ind in new_batch or ind in selected_samples:
            print("Dataoverlap")
            if break_on_overlap:
                break
        new_batch_distance.append(min_distances[ind])
        min_distances = f_dist([ind], min_distances)
        print(min_distances)
        new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.2f' % max(min_distances))
    if gpu:
        new_batch_distance, new_batch = torch.tensor(new_batch_distance).cpu().numpy().reshape(-1, 1), torch.tensor(
            new_batch).cpu().numpy()
    return new_batch_distance, new_batch


def pairwise_distances_torch(XA, XB):
    mA = XA.shape[0]
    mB = XB.shape[0]
    dm = torch.zeros((mA, mB))
    for i in range(mA):
        for j in range(mB):
            dm[i, j] = torch.pairwise_distance(XA[i], XB[j], eps=0.0)
    return dm


class CoreSetQuery(QueryMethod):
    keywords = ["CoreSet"]

    def __init__(self, method, *args, **kwargs):
        self.method=method
        self.normalize = False
        self.metric = 'euclidean'


    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs) -> Tuple[List, List]:
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_idx_set = unlabeled_idx_set if unlabeled_idx_set is not None else dataset_handler.unlabeled_idcs
        labeled_idx_set = labeled_idx_set if labeled_idx_set is not None else dataset_handler.labeled_idcs



        pool_loader = dataset_handler.get_full_pool_loader(unlabeled_idx_set, labeled_idx_set)
        valued_list, indices = get_kcg_handler(models, labeled_idx_set, unlabeled_idx_set, pool_loader, device,
                                               query_size,
                                               handler=handler, normalize=self.normalize,metric=self.metric)

        assert max(indices) < len(unlabeled_idx_set), "indices should be only in range of unlabeled idx set"
        set_indices = [unlabeled_idx_set[i] for i in indices]
        return valued_list, set_indices

class BadgeQuery(QueryMethod):
    # kmeans ++ initialization
    # import pdb
    keywords = ["Badge"]
    @staticmethod
    def init_centers(X, K):
        from sklearn.metrics import pairwise_distances
        from scipy import stats
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        unc = []
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            # if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            # normed distances
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            # drawing from a probability
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            unc.append(D2[ind])
            indsAll.append(ind)
            cent += 1
        return unc, indsAll

    @staticmethod
    def query(models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs) -> Tuple[List, List]:
        unlabeled_idx_set = unlabeled_idx_set if unlabeled_idx_set is not None else dataset_handler.unlabeled_idcs
        labeled_idx_set = labeled_idx_set if labeled_idx_set is not None else dataset_handler.labeled_idcs
        #pool_loader = dataset_handler.get_full_pool_loader(unlabeled_idx_set, labeled_idx_set)
        dataset=dataset_handler.train_pool
        gradEmbedding = handler.get_grad_embedding(dataset, models["task"], unlabeled_idx_set, dataset_handler.dataset_config,
                                           config).numpy()
        unc, chosen = BadgeQuery.init_centers(gradEmbedding, query_size)
        return unc, [unlabeled_idx_set[i] for i in chosen]


def get_kcg(models, labeled_data_idx, unlabeled_data_idx, unlabeled_loader, device, query_size):
    """
    k-center greedy
    :param models:
    :param labeled_data_idx:
    :param unlabeled_data_idx:
    :param unlabeled_loader:
    :param device:
    :param query_size:
    :return:
    """
    models["task"].eval()
    with torch.cuda.device(device):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            with torch.cuda.device(device):
                inputs = inputs.cuda()
            _, features_batch, _ = models["task"](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        # dataloader contains [unlabeled + labeled data, so new_av_idx are only the labeled indices
        labeled_av_idx = np.arange(len(unlabeled_data_idx), (len(unlabeled_data_idx) + len(labeled_data_idx)))
        sampling = kCenterGreedy(feat)
        batch_values, batch = sampling.select_batch_(labeled_av_idx, query_size)
        other_idx = [x for x in range(len(unlabeled_data_idx)) if x not in batch]
    return batch_values, batch + other_idx


def get_kcg_handler(models, labeled_data_idx, unlabeled_data_idx, unlabeled_loader, device, query_size, handler=None, normalize=False, metric='euclidean'):

    feat = handler.get_features(models, unlabeled_loader, device).detach().cpu().numpy()

    if normalize:
        feat = np.array(nn.functional.normalize(torch.tensor(feat)))
    # dataloader contains [unlabeled + labeled data, so new_av_idx are only the labeled indices
    labeled_av_idx = np.arange(len(unlabeled_data_idx), (len(unlabeled_data_idx) + len(labeled_data_idx)))
    sampling = kCenterGreedy(feat,metric=metric)
    batch_values, batch = sampling.select_batch_(labeled_av_idx, query_size)
    batch_values=[float(b) for b in batch_values]
    other_idx = [x for x in range(len(unlabeled_data_idx)) if x not in batch]
    return batch_values, batch + other_idx
