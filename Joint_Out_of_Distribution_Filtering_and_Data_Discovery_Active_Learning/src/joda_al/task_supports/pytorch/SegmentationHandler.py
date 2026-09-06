import torch
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from joda_al.utils.logging_utils.settings import get_global_verbosity
from joda_al.data_loaders.SemanticSegmentation.segmentation_dataset import to_onehot_torch
from joda_al.utils.image_creators import create_prediction_image
from joda_al.Pytorch_Extentions.metrics.cityscapes_eval_scripts.evalPixelLevelSemanticLabeling import get_cityscapes_evaluator
from joda_al.Pytorch_Extentions.metrics.streamIoU import StreamSegMetrics
from joda_al.models.loss_functions import loss_pred_loss
from joda_al.task_supports.pytorch.ClassificationHandler import ClassificationHandler
from joda_al.utils.logging_utils.training_logger import global_write_fig
from joda_al.utils.method_utils import is_loss_learning


class SegementationHandler(ClassificationHandler):
    """
    Handler for semantic segmentation tasks, providing training, evaluation, and feature extraction methods.
    Inherits from ClassificationHandler.
    """
    stats: Dict[str, Optional[Any]] = {
        "loss": None,
        "val_mIoU": None,
        "test_mIoU": None,
        "val_MIoU" : None,
        "test_MIoU": None,
        "val_cmIoU": None,
        "test_cmIoU": None,
        "val_catIoU": None,
        "test_catIoU": None,
        "valued_unlabeled_data": None,
        "selected_idx": None,
    }

    @staticmethod
    def get_softmax_score(score: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Applies softmax to the output tensor.

        Args:
            score: Dictionary containing the output tensor under key "out".

        Returns:
            Softmaxed output tensor.
        """
        return torch.softmax(score["out"], dim=-1)

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
        Performs a single training step for segmentation.

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
            Tuple of (loss, module_loss, output tensor).
        """
        results, _, loss_features = models["task"](inputs)
        target_loss = criterion(results["out"], labels)
        if "aux" in results:
            target_loss += criterion(results["aux"], labels)
        # loss per elememt:
        # loss per batch:
        # loss=torch.sum(loss_per_element) / loss_per_element.size(0)
        if is_loss_learning(method):
            lloss_weight = config["lloss_weight"]
            margin = config["margin"]
            if epoch > epoch_loss:
                for i in range(len(loss_features)):
                    loss_features[i] = loss_features[i].detach()

            pred_loss = models['module'](loss_features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss = loss_pred_loss(config["lloss_loss_function"], pred_loss, target_loss, margin=margin)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            if m_module_loss is not None:
                loss = m_backbone_loss + lloss_weight * m_module_loss
            else:
                loss = m_backbone_loss
        else:
            loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss = None
        return loss, m_module_loss, results["out"]

    @classmethod
    def evaluate(
        cls,
        device: torch.device,
        dataloader: Any,
        models: Dict[str, Any],
        method: str,
        criterion: Any,
        config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        cycle_num: int,
        mode: str
    ) -> Tuple[Dict[str, Any], float, Optional[float]]:
        """
        Evaluates the model(s) on the given dataloader for segmentation.

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
        loss_list: List[float] = []
        module_loss_list: List[float] = []
        batch_sizes: List[int] = []
        sm = StreamSegMetrics(dataset_config["num_classes"])
        if mode == "test" and dataset_config["dataset_name"] == "Cityscapes":
            city_evaluator = get_cityscapes_evaluator()
        with torch.no_grad():
            for (inputs, labels) in tqdm(
                dataloader, leave=False, total=len(dataloader), desc='Iterations',
                unit="Batch", position=1, disable=get_global_verbosity()
            ):
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                loss, module_loss, preds = cls.train_step(models, method, criterion, inputs, labels, 0, config, 999)
                loss_list.append(loss.detach().data.cpu().numpy())
                if module_loss is not None:
                    module_loss_list.append(module_loss.detach().data.cpu().numpy())
                batch_sizes.append(labels.shape[0])
                sm.update(labels.cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())
                if mode == "test" and dataset_config["dataset_name"] == "Cityscapes":
                    city_evaluator.update(preds.detach().max(dim=1)[1].cpu().numpy(), labels.cpu().numpy())
                if config["store_predictions"]:
                    store_semseg_predictions(inputs, preds, batch_sizes, dataset_config, mode, cycle_num)
                    store_semseg_predictions(inputs, labels, batch_sizes, dataset_config, mode + "label", cycle_num)
        results = sm.get_results(dataset_config["ignored_classes"])
        cmIoU = results["Mean Class IoU"]
        if mode == "test" and dataset_config["dataset_name"] == "Cityscapes":
            resultes_dict = city_evaluator.compute()
            cmIoU = resultes_dict["averageScoreClasses"]
        print(str(results))
        metric = {"mIoU": results["Mean IoU"], "MIoU": results["Micro Mean IoU"], "cmIoU": cmIoU}
        if len(module_loss_list) > 0:
            mod_loss = sum(module_loss_list) / len(module_loss_list)
        else:
            mod_loss = None
        return metric, sum(loss_list) / len(loss_list), mod_loss

    @staticmethod
    def evaluate_segmentation_ignite(
        device: torch.device,
        dataloader: Any,
        models: Dict[str, Any],
        method: str,
        criterion: Any,
        training_config: Dict[str, Any],
        dataset_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Evaluates segmentation using Ignite metrics.

        Args:
            device: CUDA device.
            dataloader: DataLoader for evaluation.
            models: Dictionary of models.
            method: Method name.
            criterion: Loss function.
            training_config: Training configuration.
            dataset_config: Dataset configuration.

        Returns:
            Tuple of (metrics dictionary, average loss).
        """
        from ignite.metrics import IoU, ConfusionMatrix, mIoU, Loss
        from ignite.engine import Engine
        with torch.no_grad():
            def eval_dl(engine, batch):
                with torch.cuda.device(device):
                    inputs = batch[0].cuda()
                    labels = batch[1].cuda()
                loss, module_loss, preds = SegementationHandler.train_step(
                    models, method, criterion, inputs, labels, 0, training_config, 999
                )
                return preds, labels

            def loss_func(results, labels):
                target_loss = criterion(results, labels)
                loss = torch.sum(target_loss) / target_loss.size(0)
                return loss

            default_evaluator = Engine(eval_dl)
            cm = ConfusionMatrix(num_classes=dataset_config["num_classes"])
            iou_metric = IoU(cm)
            miou_metric = mIoU(cm)
            mloss = Loss(loss_func)
            iou_metric.attach(default_evaluator, 'iou')
            miou_metric.attach(default_evaluator, 'miou')
            mloss.attach(default_evaluator, 'loss')

            state = default_evaluator.run(dataloader)
        print(state.metrics['iou'])
        metric = {"mIoU": state.metrics['miou'], "iou": state.metrics['iou']}
        return metric, state.metrics['loss']

    @staticmethod
    def get_features(
        models: Dict[str, Any],
        data_loader: Any,
        device: torch.device
    ) -> torch.Tensor:
        """
        Extracts features from the model for the given data loader.

        Args:
            models: Dictionary of models.
            data_loader: DataLoader for feature extraction.
            device: CUDA device.

        Returns:
            Feature tensor.
        """
        models["task"].eval()
        with torch.cuda.device(device):
            features = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, _ in data_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    output = models["task"](inputs)
                    features_batch = output[1]
                features = torch.cat((features, features_batch), 0)
            feat = features
        return feat

    @staticmethod
    def get_head_features(
        models: Dict[str, Any],
        data_loader: Any,
        device: torch.device
    ) -> torch.Tensor:
        """
        Extracts head features from the model for the given data loader.

        Args:
            models: Dictionary of models.
            data_loader: DataLoader for feature extraction.
            device: CUDA device.

        Returns:
            Feature tensor.
        """
        models["task"].eval()
        with torch.cuda.device(device):
            features = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, _ in data_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    output = models["task"](inputs)
                    features_batch = output[0]
                features = torch.cat((features, features_batch), 0)
            feat = features
        return feat


def custom_iou(
    labels: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Computes the mean Intersection over Union (IoU) for predictions and labels.

    Args:
        labels: Ground truth labels tensor.
        preds: Prediction tensor (logits or probabilities).
        num_classes: Number of classes.

    Returns:
        Mean IoU tensor.
    """
    l_h = to_onehot_torch(labels)
    pred_cls = torch.max(preds.data, dim=1).indices
    pred_cls = to_onehot_torch(pred_cls)
    intersection = torch.logical_and(l_h, pred_cls)
    union = torch.logical_or(l_h, pred_cls)
    iou_score = torch.sum(intersection, dim=[2, 3]) / (torch.sum(union, dim=[2, 3]) + 1e-4)
    return iou_score.mean(dim=-1)


def store_semseg_predictions(
    samples: torch.Tensor,
    predictions: torch.Tensor,
    batch_sizes: List[int],
    dataset_config: Dict[str, Any],
    mode: str,
    cycle_num: int
) -> None:
    """
    Stores semantic segmentation predictions as images.

    Args:
        samples: Input samples tensor.
        predictions: Predictions tensor.
        batch_sizes: List of batch sizes.
        dataset_config: Dataset configuration.
        mode: Mode string (e.g., "test", "val").
        cycle_num: Current cycle number.

    Returns:
        None
    """
    img_idx = sum(batch_sizes[:-1])
    for cur_idx, (sample, pred) in enumerate(zip(samples, predictions)):
        fig = create_prediction_image(sample.cpu().numpy(), pred.cpu().numpy(), dataset_config)
        global_write_fig(f'{mode.capitalize()}{cycle_num}/Preds', fig, img_idx + cur_idx)
        break
