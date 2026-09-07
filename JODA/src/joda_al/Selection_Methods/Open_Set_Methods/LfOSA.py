from joda_al.data_loaders.data_handler.data_handler import DataSetHandler
from joda_al.data_loaders.sampler import SubsetSequentialSampler
from joda_al.Selection_Methods.Open_Set_Methods.OpenSetQueryDef import OpenSetQueryMethod
from joda_al.Selection_Methods.query_method_def import QueryMethod
from torch.utils.data import DataLoader
from joda_al.models.model_lib import ModelHandler
from joda_al.task_supports.pytorch.TaskHandler import TaskHandler
from typing import Dict, Union, Optional, Tuple, List
import copy
import os
import numpy as np
import torch
from sklearn.mixture import GaussianMixture



class LfOSA(OpenSetQueryMethod):
    keywords = ["LfOSA"]

    def __init__(self, method, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    def query(self, models: Dict, handler: Union[TaskHandler, ModelHandler], dataset_handler: DataSetHandler,
              config: Dict, query_size: int, device,
              labeled_idx_set: Optional[int] = None,
              unlabeled_idx_set: Optional[int] = None,
              *arg, **kwargs):
        models["module"].eval()
        queryIndex = []
        labelArr = []
        uncertaintyArr = []
        S_ij = {}
        self.num_classes = int(list(models["task"].eval().children())[-1].out_features)

        Len_labeled_ind_train = len(labeled_idx_set)
        unlabeledloader = dataset_handler.get_unlabeled_pool_loader(unlabeled_idx_set)

        for batch_idx, (data, labels) in enumerate(unlabeledloader):
            if True:
                data, labels = data.cuda(), labels.cuda()
            outputs, _, _ = models["module"](data)
            index = (int(batch_idx * unlabeledloader.batch_size) + torch.arange(len(data))).tolist()
            queryIndex += index
            labelArr += list(np.array(labels.cpu().data))
            # activation value based
            v_ij, predicted = outputs.max(1)
            for i in range(len(predicted.data)):
                tmp_class = np.array(predicted.data.cpu())[i]
                tmp_index = index[i]
                tmp_label = np.array(labels.data.cpu())[i]
                tmp_value = np.array(v_ij.data.cpu())[i]
                if tmp_class not in S_ij:
                    S_ij[tmp_class] = []
                S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])


        # fit a two-component GMM for each class
        tmp_data = []
        for tmp_class in S_ij:
            S_ij[tmp_class] = np.array(S_ij[tmp_class])
            activation_value = S_ij[tmp_class][:, 0]
            if len(activation_value) < 2:
                continue
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(np.array(activation_value).reshape(-1, 1))
            prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
            # 得到为known类别的概率
            prob = prob[:, gmm.means_.argmax()]
            # 如果为unknown类别直接为0
            if tmp_class == self.num_classes:
                prob = [0]*len(prob)
                prob = np.array(prob)

            if len(tmp_data) == 0:
                tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
            else:
                tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))


        tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
        tmp_data = tmp_data.T
        queryIndex = tmp_data[2][-query_size:].astype(int)
        labelArr = tmp_data[3].astype(int)
        queryLabelArr = tmp_data[3][-query_size:]
        precision = len(np.where(queryLabelArr < self.num_classes)[0]) / len(queryLabelArr)
        recall = (len(np.where(queryLabelArr < self.num_classes)[0]) + Len_labeled_ind_train) / (
                len(np.where(labelArr < self.num_classes)[0]) + Len_labeled_ind_train)
        
        set_indices = [unlabeled_idx_set[i] for i in queryIndex]
        values = tmp_data[0][-query_size:]
        return values, set_indices