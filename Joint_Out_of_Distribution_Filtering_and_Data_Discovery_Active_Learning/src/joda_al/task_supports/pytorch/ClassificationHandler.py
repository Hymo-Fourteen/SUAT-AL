from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from joda_al.models.model_lib import EnsembleModel
from joda_al.utils.logging_utils.settings import get_global_verbosity
from joda_al.data_loaders.sampler import SubsetSequentialSampler
from joda_al.defintions import DataUpdateScenario
from joda_al.models.loss_functions import LOSS_FUNCTIONS, loss_pred_loss
from joda_al.task_supports.pytorch.TaskHandler import DefaultTaskHandler
from joda_al.utils.logging_utils.log_writers import gl_info
from joda_al.utils.method_utils import is_loss_learning


class ClassificationHandler(DefaultTaskHandler):
    """
    Handler for classification tasks, providing training, evaluation, and utility methods
    for active learning and open set scenarios.
    """

    @classmethod
    def get_stats(cls, scenario: Any) -> Dict[str, Optional[Any]]:
        """
        Returns a dictionary of statistics keys initialized to None, depending on the scenario.

        Args:
            scenario: The data update scenario.

        Returns:
            Dictionary with statistics keys and None values.
        """
        if scenario == DataUpdateScenario.osal_extending:
            return {
                "loss": None,
                "val_acc": None,
                "test_acc": None,
                "test_f1": None,
                "test_avg_acc": None,
                "valued_unlabeled_data": None,
                "selected_idx": None,
            }
        else:
            return {
                "loss": None,
                "val_acc": None,
                "test_acc": None,
                "test_f1": None,
                "valued_unlabeled_data": None,
                "selected_idx": None,
            }

    @staticmethod
    def get_grad_embedding(
        dataset: Any,
        model: Any,
        unlabeled_idx_set: Sequence[int],
        dataset_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Computes gradient embeddings for BADGE active learning.

        Args:
            dataset: Dataset object.
            model: Model with get_embedding_dim() and forward method.
            unlabeled_idx_set: Indices of unlabeled data.
            dataset_config: Dataset configuration dictionary.
            training_config: Training configuration dictionary.

        Returns:
            torch.Tensor: Gradient embeddings of shape (num_samples, embDim * num_classes).
        """
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = dataset_config["num_classes"]
        embedding = np.zeros([len(unlabeled_idx_set), embDim * nLab])
        unlabeled_loader = DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            sampler=SubsetSequentialSampler(unlabeled_idx_set),
            pin_memory=True
        )
        with torch.no_grad():
            for idx, (x, y) in enumerate(unlabeled_loader):
                x, y = x.cuda(), y.cuda()
                outpred, emb, _ = model(x)
                emb = emb.cpu().numpy()
                outpred = outpred.cpu().numpy()
                maxInds = np.argmax(outpred, 1)
                for j in range(len(y)):
                    b_idx = training_config["batch_size"] * idx + j
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[b_idx][embDim * c: embDim * (c + 1)] = deepcopy(emb[j]) * (1 - outpred[j][c])
                        else:
                            embedding[b_idx][embDim * c: embDim * (c + 1)] = deepcopy(emb[j]) * (-1 * outpred[j][c])
        return torch.Tensor(embedding)

    @staticmethod
    def setup_loss_function(
        method: str,
        training_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        task: str,
        cls_weights: Optional[torch.Tensor],
        device: torch.device
    ) -> Any:
        """
        Sets up the loss function for training.

        Args:
            method: Name of the method.
            training_config: Training configuration dictionary.
            dataset_config: Dataset configuration dictionary.
            task: Task name.
            cls_weights: Optional class weights tensor.
            device: Device to use.

        Returns:
            Loss function instance.

        Raises:
            NotImplementedError: If the loss function is not implemented.
        """
        if training_config["cls_weights"] == "all" and cls_weights is not None:
            print(f"Using weights: {cls_weights}")
            weights = cls_weights.to(device)
        else:
            weights = None
        if training_config["loss_func"] in ["CrossEntropy", "GorLoss"]:
            criterion = LOSS_FUNCTIONS[task].get(training_config["loss_func"], None)(weight=weights, reduction='none')
        else:
            criterion = LOSS_FUNCTIONS[task].get(training_config["loss_func"], None)(
                weight=weights, reduction='none', **training_config, **dataset_config
            )
        if criterion is None:
            raise NotImplementedError(f'Loss function {training_config["loss_func"]} not implemented')
        return criterion

    @staticmethod
    def train_step(
        models: Dict[str, Any],
        method: str,
        criterion: Any,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        config: Dict[str, Any],
        epoch_loss: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Performs a single training step.

        Args:
            models: Dictionary of models.
            method: Method name.
            criterion: Loss function.
            inputs: Input tensor.
            labels: Label tensor.
            epoch: Current epoch.
            config: Training configuration.
            epoch_loss: Epoch after which to detach features.

        Returns:
            Tuple of (loss, module_loss, predictions).
        """
        if config.get("ensemble", False) is True:
            for model in models["task"]:
                scores, _, features = model(inputs)
                target_loss = criterion(scores, labels)
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                loss = m_backbone_loss
                _, preds = torch.max(scores.data, 1)
            m_module_loss = None
        else:
            scores, _, features = models["task"](inputs)
            target_loss = criterion(scores, labels)

            if not config["softmax"]:
                _, preds = torch.max(torch.softmax(scores.data, dim=1), 1)
            else:
                _, preds = torch.max(scores.data, 1)
            if is_loss_learning(method):
                lloss_weight = config["lloss_weight"]
                margin = config["margin"]
                lloss_reg = config["lloss_reg"]
                if epoch > epoch_loss:
                    gl_info("Deteched Loss")
                    gl_info(epoch_loss)
                    for i in range(len(features)):
                        features[i] = features[i].detach()

                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                m_module_loss = loss_pred_loss(
                    config["lloss_loss_function"], pred_loss, target_loss, margin=margin, reg=lloss_reg
                )
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                if m_module_loss is not None:
                    loss = m_backbone_loss + lloss_weight * m_module_loss
                else:
                    loss = m_backbone_loss
            else:
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                loss = m_backbone_loss
                m_module_loss = None
        return loss, m_module_loss, preds

    @staticmethod
    def train_step_OpenSet(
        models: Dict[str, Any],
        method: str,
        criterion: Any,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mode: str = "train"
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Performs a training step for open set methods.

        Args:
            models: Dictionary of models.
            method: Method name.
            criterion: Loss function.
            inputs: Input tensor.
            labels: Label tensor.
            mode: Mode string ("train" or "eval").

        Returns:
            Tuple of (classifier_loss, detector_loss, predictions, iD_labels).
        """
        classifier_loss, detector_loss, classifier_outputs, iD_labels = criterion(models, inputs, labels, mode=mode)
        if isinstance(classifier_outputs, list):
            preds = []
            for out in classifier_outputs:
                _, pred = torch.max(out.data, 1)
                preds.append(pred)
        else:
            _, preds = torch.max(classifier_outputs.data, 1)
        return classifier_loss, detector_loss, preds, iD_labels

    @classmethod
    def evaluate(
        cls,
        device: torch.device,
        dataloader: DataLoader,
        models: Dict[str, Any],
        method: str,
        criterion: Any,
        config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        cycle_num: int,
        mode: str
    ) -> Tuple[Dict[str, Any], float, Optional[float]]:
        """
        Evaluates the model(s) on the given dataloader.

        Args:
            device: CUDA device.
            dataloader: DataLoader for evaluation.
            models: Dictionary of models.
            method: Method name.
            criterion: Loss function.
            config: Training configuration.
            dataset_config: Dataset configuration.
            cycle_num: Current cycle number.
            mode: Evaluation mode.

        Returns:
            Tuple of (metrics dictionary, average loss, average module loss).
        """
        total = 0
        if config.get("ensemble", False) is True:
            correct: Union[List[int], int] = []
        else:
            correct = 0
        loss_list: List[torch.Tensor] = []
        module_loss_list: List[Any] = []
        with torch.cuda.device(device):
            all_preds = torch.tensor([]).cuda()
            all_labels = torch.tensor([]).cuda()

        with torch.no_grad():
            for (inputs, labels) in tqdm(
                dataloader, leave=False, total=len(dataloader), desc='Iterations',
                unit="Batch", position=1, disable=get_global_verbosity()
            ):
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                if config["loss_func"] in ["OODCrossEntropy", "OpenCrossEntropy", "OutlierExposure", "EnergyExposure"]:
                    loss, module_loss, preds, labels = cls.train_step_OpenSet(
                        models, method, criterion, inputs, labels, mode="eval"
                    )
                    if isinstance(loss, list):
                        loss = torch.mean(torch.stack(loss))
                else:
                    loss, module_loss, preds = cls.train_step(
                        models, method, criterion, inputs, labels, 0, config, epoch_loss=config["epoch_loss"]
                    )

                loss_list.append(loss.cpu())
                if module_loss is not None:
                    module_loss_list.append(module_loss)
                if method in ["AGD"]:
                    inD_indices = (labels >= 0) & (labels < len(config["ind_classes"]))
                    mask_lab = inD_indices.nonzero(as_tuple=True)[0]
                    labels = labels[mask_lab]
                    if len(mask_lab) == 0:
                        continue

                if config.get("ensemble", False) is True:
                    for i in range(len(preds)):
                        correct.append((preds[i] == labels[i]).sum().item())
                    total += labels[0].size(0)
                    preds = torch.concat([torch.unsqueeze(p, dim=1) for p in preds], dim=1)
                    labels = torch.concat([torch.unsqueeze(l, dim=1) for l in labels], dim=1)
                else:
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                all_preds = torch.cat((all_preds, preds.to(all_preds.dtype)), 0)
                all_labels = torch.cat((all_labels, labels.to(all_labels.dtype)), 0)

        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()
        if config.get("ensemble", False) is True:
            ens_all_preds = all_preds_np
            ens_all_labels = all_labels_np
            all_labels_np = all_labels_np[:, 0]
            all_preds_np = all_preds_np[:, 0]
            ens_correct = np.array(correct)
            if not isinstance(models["task"], EnsembleModel):
                correct = sum(correct)
            else:
                correct = correct[0]
        precision = precision_score(all_labels_np, all_preds_np, average="weighted")
        recall = recall_score(all_labels_np, all_preds_np, average="weighted")
        f1 = f1_score(all_labels_np, all_preds_np, average="weighted")
        if config.get("data_scenario", "") == DataUpdateScenario.osal_extending:
            extending_accuracy = (dataset_config["num_classes"] / dataset_config["all_classes"]) * balanced_accuracy_score(all_labels_np, all_preds_np)
            metric = {"acc": correct / total, "f1": f1, "precision": precision, "recall": recall, "avg_acc": extending_accuracy}
        else:
            metric = {"acc": correct / total, "f1": f1, "precision": precision, "recall": recall}
        if isinstance(models["task"], EnsembleModel):
            for i in range(ens_all_preds.shape[1]):
                precision = precision_score(ens_all_labels[:, i], ens_all_preds[:, i], average="weighted")
                recall = recall_score(ens_all_labels[:, i], ens_all_preds[:, i], average="weighted")
                f1 = f1_score(ens_all_labels[:, i], ens_all_preds[:, i], average="weighted")
                metric[f"acc_{i}"] = ens_correct[i] / total
                metric[f"f1_{i}"] = f1
                metric[f"precision_{i}"] = precision
                metric[f"recall_{i}"] = recall
                if config.get("data_scenario", "") == DataUpdateScenario.osal_extending:
                    extending_accuracy = (dataset_config["num_classes"] / dataset_config["all_classes"]) * balanced_accuracy_score(ens_all_labels[:, i], ens_all_preds[:, i])
                    metric[f"avg_acc_{i}"] = extending_accuracy
        if len(module_loss_list) > 0:
            mod_loss = sum(module_loss_list) / len(module_loss_list)
        else:
            mod_loss = None
        return metric, sum(loss_list) / len(loss_list), mod_loss

