from abc import abstractmethod
from typing import Dict
import torch


class TaskHandler():
    """
    Abstract base class for handling tasks in a machine learning pipeline.
    Provides static methods for setting up loss functions, performing training steps,
    and extracting features or predictions from models.
    """
    pass
    @staticmethod
    def setup_loss_function(method, training_config, dataset_config, task, cls_weights, device):
        """
        Set up the loss function for the given method and task.

        Args:
            method: The method to use for training.
            training_config: Configuration for the training process.
            dataset_config: Configuration for the dataset.
            task: The task being performed (e.g., classification, segmentation).
            cls_weights: Class weights for the loss function.
            device: The device (CPU/GPU) to use for computation.
        """
        pass

    @staticmethod
    @abstractmethod
    def train_step(models: Dict[str, torch.nn.Module], method, criterion, inputs, labels, epoch, training_config, epoch_loss):
        """
        Perform a single training step.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used in training.
            method: The training method.
            criterion: The loss function.
            inputs: Input data for the training step.
            labels: Ground truth labels for the input data.
            epoch: Current epoch number.
            training_config: Configuration for the training process.
            epoch_loss: Accumulated loss for the current epoch.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_features(models, data_loader, device,config=None):
        """
        Extract features from the models for the given data loader.

        Args:
            models: Dictionary of models used for feature extraction.
            data_loader: DataLoader providing the input data.
            device: The device (CPU/GPU) to use for computation.
            config: Optional configuration for feature extraction.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_features_with_labels(models, data_loader, device,config=None):
        """
        Extract features and corresponding labels from the models for the given data loader.

        Args:
            models: Dictionary of models used for feature extraction.
            data_loader: DataLoader providing the input data and labels.
            device: The device (CPU/GPU) to use for computation.
            config: Optional configuration for feature extraction.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_features_with_predictions(models, data_loader, device,config=None):
        pass


class DefaultTaskHandler(TaskHandler):

    stats={}

    @classmethod
    def get_stats(cls, scenario):
        """
        Retrieve the stored statistics for a given scenario.

        Args:
            scenario: The scenario for which statistics are requested.

        Returns:
            dict: The stored statistics.
        """
        return cls.stats

    fn_collate=None
    """
    Placeholder for a custom collate function, if needed.
    """

    @staticmethod
    def get_predicted_loss(models, unlabeled_loader, device):
        """
        Compute the predicted loss for unlabeled data.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used for prediction.
            unlabeled_loader: DataLoader providing the unlabeled data.
            device: The device (CPU/GPU) to use for computation.

        Returns:
            numpy.ndarray: Array of predicted losses for the unlabeled data.
        """
        models["task"].eval()
        models['module'].eval()
        with torch.cuda.device(device):
            uncertainty = torch.tensor([]).cuda()

        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                _, _, features = models["task"](inputs)
                pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))
                uncertainty = torch.cat((uncertainty, pred_loss), 0)

        return uncertainty.cpu().numpy()

    @staticmethod
    def get_features(models, data_loader, device,config=None):
        """
        Extract features from the models for the given data loader.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used for feature extraction.
            data_loader: DataLoader providing the input data.
            device: The device (CPU/GPU) to use for computation.
            config: Optional configuration for feature extraction.

        Returns:
            torch.Tensor: Extracted features.
        """
        models["task"].eval()
        with torch.cuda.device(device):
            features = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, _ in data_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    output = models["task"](inputs)
                    #: scores, features, features_unflat
                    # NPN: scores, features, features_unflat, features_flow
                    features_batch = output[1]
                features = torch.cat((features, features_batch), 0)
            feat = features  # .detach().cpu().numpy()
        return feat

    @staticmethod
    def get_head_features(models, data_loader, device):
        """
        Extract head features (e.g., scores) from the models for the given data loader.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used for feature extraction.
            data_loader: DataLoader providing the input data.
            device: The device (CPU/GPU) to use for computation.

        Returns:
            torch.Tensor: Extracted head features.
        """
        models["task"].eval()
        with torch.cuda.device(device):
            features = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, _ in data_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    output = models["task"](inputs)
                    #: scores, features, features_unflat
                    # NPN: scores, features, features_unflat, features_flow
                    features_batch = output[0]
                features = torch.cat((features, features_batch), 0)
            feat = features  # .detach().cpu().numpy()
        return feat

    @staticmethod
    def get_features_with_predictions(models, data_loader, device,config=None):
        """
        Extract features and predictions from the models for the given data loader.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used for feature extraction.
            data_loader: DataLoader providing the input data.
            device: The device (CPU/GPU) to use for computation.
            config: Optional configuration for feature extraction.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Extracted features and predictions.
        """
        models["task"].eval()
        with torch.cuda.device(device):
            features = torch.tensor([]).cuda()
            predictions = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, _ in data_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    output = models["task"](inputs)
                    #: scores, features, features_unflat
                    # NPN: scores, features, features_unflat, features_flow
                    features_batch = output[1]
                    scores = output[0]
                    _, preds = torch.max(scores.data, 1)
                features = torch.cat((features, features_batch), 0)
                predictions = torch.cat((predictions, preds), 0)
            feat = features  # .detach().cpu().numpy()
        return feat, predictions

    @staticmethod
    def get_features_with_labels(models, data_loader, device,config=None):
        """
        Extract features and labels from the models for the given data loader.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used for feature extraction.
            data_loader: DataLoader providing the input data and labels.
            device: The device (CPU/GPU) to use for computation.
            config: Optional configuration for feature extraction.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Extracted features and labels.
        """
        models["task"].eval()
        with torch.cuda.device(device):
            features = torch.tensor([]).cuda()
            predictions = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, label in data_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    label = label.cuda()
                    output = models["task"](inputs)
                    #: scores, features, features_unflat
                    # NPN: scores, features, features_unflat, features_flow
                    features_batch = output[1]
                features = torch.cat((features, features_batch), 0)
                predictions = torch.cat((predictions, label), 0)
            feat = features  # .detach().cpu().numpy()
        return feat, predictions

    @staticmethod
    def get_flow_features(models, data_loader, device):
        """
        Extract flow features from the models for the given data loader.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used for feature extraction.
            data_loader: DataLoader providing the input data.
            device: The device (CPU/GPU) to use for computation.

        Returns:
            numpy.ndarray: Extracted flow features.
        """
        models["task"].eval()
        with torch.cuda.device(device):
            features = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, _ in data_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    (prediction, posterior, log_prob, flow_features), flat_features, loss_features = models["task"](inputs)
                    #: scores, features, features_unflat
                    # NPN: scores, features, features_unflat, features_flow
                    features_batch = flow_features
                features = torch.cat((features, features_batch), 0)
            feat = features.detach().cpu().numpy()
        return feat

    @staticmethod
    def get_softmax_score(score):
        """
        Extract scores from the models for the given data loader.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used for score extraction.
            data_loader: DataLoader providing the input data.
            device: The device (CPU/GPU) to use for computation.

        Returns:
            torch.Tensor: Extracted scores.
        """
        return torch.softmax(score, dim=1)

    @staticmethod
    def get_scores(models, data_loader, device):
        """
        Extract scores from the models for the given data loader.

        Args:
            models (Dict[str, torch.nn.Module]): Dictionary of models used for score extraction.
            data_loader: DataLoader providing the input data.
            device: The device (CPU/GPU) to use for computation.

        Returns:
            torch.Tensor: Extracted scores.
        """
        models["task"].eval()
        with torch.cuda.device(device):
            scores = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, _ in data_loader:
                with torch.cuda.device(device):
                    inputs = inputs.cuda()
                    score = models["task"](inputs)[0]
                    #: scores, features, features_unflat
                    # NPN: scores, features, features_unflat, features_flow
                scores = torch.cat((scores, score), 0)
            scores = scores.detach()  # .detach().cpu().numpy()
        return scores

    @staticmethod
    def unlabeled_input_wrapper(dataloader):
        """
        Wrap the input data from the dataloader for unlabeled data.

        Args:
            dataloader: DataLoader providing the unlabeled data.

        Yields:
            torch.Tensor: Input data from the dataloader.
        """
        for inputs, _ in dataloader:
            yield inputs


