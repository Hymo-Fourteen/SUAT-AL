import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np
from joda_al.defintions import TaskDef
from joda_al.models.SemanticSegmentation.loss_functions import JaccardLoss, DiceLoss
from joda_al.models.model_lib import EnsembleModel
from joda_al.utils.method_utils import is_method_of
from joda_al.utils.logging_utils.training_logger import global_write_scalar
import copy

mseloss=nn.MSELoss()


class MarginL1Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, margin, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        self.margin=margin
        super(MarginL1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = F.l1_loss(input, target, reduction="none")
        loss[loss<=self.margin]=0.0
        # print(loss)
        return loss.mean()


def loss_pred_loss(lloss_loss_function,input, target, margin=1.0, reduction='mean',reg=0.5):
    if lloss_loss_function == "original":
        return LossPredOriginalLoss(input, target, margin=margin, reduction=reduction)
    elif lloss_loss_function == "mse":
        return mseloss(input, target)
    elif lloss_loss_function == "TPL":
        return LossPredTPL(input, target, margin=margin, reduction=reduction, reg_scale=reg)
    elif lloss_loss_function == "Dist-mse":
        return LossPredDistMse(input,target, margin=margin, reduction=reduction)
    elif lloss_loss_function == "ta-vaal":
        return LossPredLoss(input, target, margin=margin, reduction=reduction)
    else:
        raise NotImplementedError(f"Loss function {lloss_loss_function} is not implemented.")


# Loss Prediction Loss for TA-VAAL
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    """
    Loss Prediction Loss as defined in the paper TA-VAAL
    """
    if len(input) <2:
        return None
    elif len(input) % 2 != 0:
        input,target=input[:-1],target[:-1]
    else:
        input,target=input,target
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def LossPredDistMse(input, target, margin=1.0, reduction='mean'):
    if len(input) <2:
        return None
    elif len(input) % 2 != 0:
        input,target=input[:-1],target[:-1]
    else:
        input,target=input,target
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.MSELoss()
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    if reduction == 'mean':
        loss = criterion(input, target)
    elif reduction == 'none':
        loss = criterion(input, target)
    else:
        NotImplementedError()
    # print(loss)
    return 2*loss


step=0


def LossPredTPL(input, target, margin=1.0, reduction='mean', reg_scale=0.5):
    """
    Loss Prediction Loss as defined in the paper TPL
    """
    global step
    if len(input) < 2:
        return None
    elif len(input) % 2 != 0:
        input, target = input[:-1], target[:-1]
    else:
        input, target = input, target
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    div = MarginL1Loss(margin)
    input_dist = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = target.detach()
    target_dist = (target - target.flip(0))[:len(target) // 2]

    diff = torch.sigmoid(input_dist)
    one = 2 * torch.sign(torch.clamp(target_dist, min=0)) - 1  # 1 operation which is defined by the authors
    if reduction == 'mean':
        kld= reg_scale*div(input,target)
        global_write_scalar('Loss/Iteration/L1', kld, step)
        step+=1
        loss = torch.sum(torch.clamp(margin - one * input_dist, min=0))
        global_write_scalar('Loss/Iteration/Margin', loss / input_dist.size(0), step)

        loss = loss / input_dist.size(0) + kld # Note that the size of input is already halved

    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


# Loss Prediction Loss
def LossPredOriginalLoss(input, target, margin=1.0, reduction='mean'):
    """
    Original Loss Prediction Loss as defined in the paper Loss Learning for Active Learning Paper
    """
    if len(input) <2 :
        return None
    elif len(input) % 2 != 0:
        input,target=input[:-1],target[:-1]
    else:
        input,target=input,target
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


def CrossEntropySeg(*args, **kwargs):
    ce= nn.CrossEntropyLoss(*args, **kwargs)
    def loss_func(input: Tensor, target: Tensor):
        loss=ce(input,target)
        loss_per_element = torch.sum(loss, dim=[1, 2]) / (loss.size(1) * loss.size(2))
        return loss_per_element
    return loss_func


def JaccardCE(*args, **kwargs):
    ce= nn.CrossEntropyLoss(*args, **kwargs)
    jar=JaccardLoss(*args, **kwargs)
    def loss_func(input: Tensor, target: Tensor):
        ce_loss=ce(input,target)
        ce_loss_per_element = torch.sum(ce_loss, dim=[1, 2]) / (ce_loss.size(1) * ce_loss.size(2))
        loss_per_element=ce_loss_per_element+jar(input,target)
        return loss_per_element
    return loss_func

class LogitNormLoss(nn.Module):
    def __init__(self, tau=0.04, weight=None, reduction='none', **kwargs):
        super(LogitNormLoss, self).__init__()
        self.tau = tau
        self.reduction = reduction
        self.weight = weight

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.tau
        loss = F.cross_entropy(logit_norm, target, reduction=self.reduction, weight=self.weight)
        return loss


class OODCrossEntropy(nn.Module):
    """
    Used in LfOSA
    """
    def __init__(self, weight=None, reduction='none', **kwargs):
        super(OODCrossEntropy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reduction = reduction
        self.weight = weight
        self.num_classes = kwargs.get("num_classes", 10)
        self.weight_cent = kwargs.get("weight_cent", 0)
        #self.criterion_cent = CenterLoss(num_classes=self.num_classes, feat_dim=2, use_gpu=True)
        self.criterion_xent = nn.CrossEntropyLoss(weight=weight, reduction="mean")
        self.triplet_mode = kwargs.get("triplet_mode", "off")

    def forward(self, model, inputs, labels, mode):
        inD_indices = (labels >= 0) & (labels < self.num_classes)
        inD_indices = inD_indices.nonzero(as_tuple=True)[0]
        if self.triplet_mode == "off":
            modified_ood_indices = (labels >= self.num_classes).nonzero(as_tuple=True)[0]
        elif self.triplet_mode == "triplet":
            modified_ood_indices = (labels < 0).nonzero(as_tuple=True)[0]
        pseudo_labels = copy.deepcopy(labels)
        pseudo_labels[modified_ood_indices] = self.num_classes
        #nearOOD_indices = (labels == -1).nonzero(as_tuple=True)[0]
        #farOOD_indices = (labels == -2).nonzero(as_tuple=True)[0]
        if mode == "train":
            if len(inD_indices) > 0 or len(modified_ood_indices) > 0:
                detector_outputs, detector_features, _ = model["module"](inputs[torch.cat((inD_indices, modified_ood_indices))].to(self.device))
                detector_outputs = detector_outputs / 0.5
                detector_xent = self.criterion_xent(detector_outputs, pseudo_labels[torch.cat((inD_indices, modified_ood_indices))])
                #detector_cent = self.criterion_cent(detector_features, labels)
                #detector_cent *= self.weight_cent
                detector_loss = detector_xent
            else:
                detector_loss = None

        else:
            detector_loss = None


        classifier_outputs, classifier_features, _ = model["task"](inputs.to(self.device))

        if len(inD_indices) > 0:
            classifier_xent = self.criterion_xent(classifier_outputs[inD_indices], labels[inD_indices])
        #classifier_cent = self.criterion_cent(classifier_features, labels[inD_indices])
        #classifier_cent *= self.weight_cent

            classifier_loss = classifier_xent
        else:
            classifier_loss = torch.tensor(0.0).to(self.device)
        return classifier_loss, detector_loss, classifier_outputs[inD_indices], labels[inD_indices]

class OpenCrossEntropy(nn.Module):
    """
    AOL Method
    """
    def __init__(self, weight=None, reduction='none', **kwargs):
        super(OpenCrossEntropy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reduction = reduction
        self.weight = weight
        self.num_classes = kwargs.get("num_classes", 10)
        self.criterion_xent = nn.CrossEntropyLoss(weight=weight, reduction="mean")
        self.triplet_mode = kwargs.get("triplet_mode", "off")

    def forward(self, model, inputs, labels, mode):
        inD_indices = (labels >= 0) & (labels < self.num_classes)
        inD_indices = inD_indices.nonzero(as_tuple=True)[0]
        if self.triplet_mode == "off":
            modified_ood_indices = (labels >= self.num_classes).nonzero(as_tuple=True)[0]
        elif self.triplet_mode == "triplet":
            modified_ood_indices = (labels < 0).nonzero(as_tuple=True)[0]
        pseudo_labels = copy.deepcopy(labels)
        pseudo_labels[modified_ood_indices] = self.num_classes
        train_idc= torch.cat((inD_indices, modified_ood_indices))
        classifier_loss=[]
        classifier_loss_outputs=[]
        classifier_labels = []
        if isinstance(model["task"],EnsembleModel):
            for m in model["task"]:
                classifier_outputs, classifier_features, _ = m(inputs.to(self.device))

                classifier_xent = self.criterion_xent(classifier_outputs[train_idc], pseudo_labels[train_idc], )
                classifier_loss.append(classifier_xent)
                classifier_loss_outputs.append(classifier_outputs[inD_indices])
                classifier_labels.append(labels[inD_indices])
        else:
            classifier_outputs, classifier_features, _ = model["task"](inputs.to(self.device))

            classifier_xent = self.criterion_xent(classifier_outputs[train_idc], pseudo_labels[train_idc],)
            classifier_loss.append(classifier_xent)
            classifier_loss_outputs.append(classifier_outputs[inD_indices])
            classifier_labels.append(labels[inD_indices])
        detector_loss = None
        return classifier_loss, detector_loss, classifier_loss_outputs, classifier_labels

class OutlierExposure(nn.Module):
    """
    Used in Joda
    """
    def __init__(self, weight=None, reduction='none', **kwargs):
        super(OutlierExposure, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reduction = reduction
        self.weight = weight
        self.num_classes = kwargs.get("num_classes", 10)
        self.classification_criterion = nn.CrossEntropyLoss(weight=weight, reduction="mean")
        self.lambda_oe = kwargs.get("lambda_oe", 0.5)
        self.triplet_mode = kwargs.get("triplet_mode", "off")

    def forward(self, model, inputs, labels, mode):
        inD_indices = (labels >= 0) & (labels < self.num_classes)
        inD_indices = inD_indices.nonzero(as_tuple=True)[0]
        outputs, features, _ = model["task"](inputs.to(self.device))
        if len(inD_indices) >0:
            loss = self.classification_criterion(outputs[inD_indices], labels[inD_indices])
        else:
            loss = torch.tensor(0.0).to(self.device)
        loss_oe = torch.tensor(0.0).to(self.device)
        if mode == "train":
            if self.triplet_mode == "off":
                ood_indices = (labels >= self.num_classes).nonzero(as_tuple=True)[0]
            elif self.triplet_mode == "triplet":
                ood_indices = (labels < 0).nonzero(as_tuple=True)[0]
            loss_oe = -(
                outputs[ood_indices].mean(1) -
                torch.logsumexp(outputs[ood_indices], dim=1)).mean()
            #-(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
            if len(ood_indices)> 0:
                loss += self.lambda_oe * loss_oe
        return loss, loss_oe, outputs[inD_indices], labels[inD_indices]

class EnergyExposure(nn.Module):
    """
    Used in Joda
    """
    def __init__(self, weight=None, reduction='none', **kwargs):
        super(EnergyExposure, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reduction = reduction
        self.weight = weight
        self.num_classes = kwargs.get("num_classes", 10)
        self.classification_criterion = nn.CrossEntropyLoss(weight=weight, reduction="mean")
        self.lambda_oe = kwargs.get("lambda_oe", 0.1)
        self.triplet_mode = kwargs.get("triplet_mode", "off")
        self.margin_out=kwargs.get("margin_out", -5)
        self.margin_in = kwargs.get("margin_in", -25)

    def forward(self, model, inputs, labels, mode):
        inD_indices = (labels >= 0) & (labels < self.num_classes)
        inD_indices = inD_indices.nonzero(as_tuple=True)[0]
        outputs, features, _ = model["task"](inputs.to(self.device))
        if len(inD_indices) >0:
            loss = self.classification_criterion(outputs[inD_indices], labels[inD_indices])
        else:
            loss = torch.tensor(0.0).to(self.device)
        loss_oe = torch.tensor(0.0).to(self.device)
        if mode == "train":
            if self.triplet_mode == "off":
                ood_indices = (labels >= self.num_classes).nonzero(as_tuple=True)[0]
            elif self.triplet_mode == "triplet":
                ood_indices = (labels < 0).nonzero(as_tuple=True)[0]
            elif self.triplet_mode == "triplet2":
                ood_indices = (labels < 0).nonzero(as_tuple=True)[0]
            Ec_out = -torch.logsumexp(outputs[ood_indices], dim=1)
            loss_e_out = torch.pow(F.relu(Ec_out-self.margin_out), 2).mean()
            loss_oe+=loss_e_out
            if self.triplet_mode == "triplet2":
                Ec_in = -torch.logsumexp(outputs[inD_indices], dim=1)
                loss_e_in = torch.pow(F.relu(Ec_in-self.margin_in), 2).mean()
                loss_oe+=loss_e_in
            if len(ood_indices)> 0:
                loss += self.lambda_oe * loss_oe
        return loss, loss_oe, outputs[inD_indices], labels[inD_indices]


class DistillLoss(nn.Module):
    """
    https://github.com/mashijie1028/ActiveGCD/blob/main/model.py#L237
    MIT license
    Used in Active General Discovery Learning
    """
    def __init__(self, warmup_teacher_temp_epochs, nepochs,
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class SupConLoss(torch.nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast
    https://github.com/mashijie1028/ActiveGCD/blob/main/model.py#L237
    MIT license
    Used in Active General Discovery Learning
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def update_ema_variables(model, ema_model, alpha):
    # Use the true average until the exponential average is more correct
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):
    """
    https://github.com/mashijie1028/ActiveGCD/blob/main/model.py#L237
    MIT license
    :param features: 
    :param n_views: 
    :param temperature: 
    :param device: 
    :return: 
    """

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def ova_loss(logits_open, label):
    """
    Apache-2.0 license
    https://github.com/njustkmg/NeurIPS2023-PAL/blob/main/utils/misc.py
    Used in PAL
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    open_l = torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1)
    open_l_neg = torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0]
    Lo = open_loss_neg + open_loss
    return Lo

def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    L_c = torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1)
    return Le, L_c

LOSS_FUNCTIONS={
    TaskDef.classification:{
    "CrossEntropy": nn.CrossEntropyLoss,
    "LogitNorm": LogitNormLoss,
    "GorLoss": nn.CrossEntropyLoss,
    "OutlierExposure": OutlierExposure,
   "EnergyExposure": EnergyExposure,
    "OODCrossEntropy": OODCrossEntropy,
    "OpenCrossEntropy": OpenCrossEntropy,
    },
    TaskDef.semanticSegmentation: {
        "CrossEntropy": CrossEntropySeg,
        "Dice": DiceLoss,
        "Jaccard": JaccardLoss,
        "JaccardCE": JaccardCE
    },
    TaskDef.objectDetection: {
        "Model_Defined": None
    }
}