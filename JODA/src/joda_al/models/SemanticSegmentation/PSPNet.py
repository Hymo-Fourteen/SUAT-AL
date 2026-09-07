from typing import Optional

import torch
from torch import nn
from torchvision.models.segmentation.fcn import FCNHead
from joda_al.models import torchResnet as resnet
from joda_al.models import torchMobilenet as mobilenetv3
from joda_al.models.SemanticSegmentation.deeplabV3 import _SimpleSegmentationModel
from torch.nn import functional as F

from joda_al.models.model_lib import McDropoutModelMixin


class PSPNet(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PSPBlock(
                    in_channels,
                    in_channels // len(sizes),
                    size,
                    use_bathcnorm=use_bathcnorm,
                )
                for size in sizes
            ]
        )

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module, McDropoutModelMixin):
    def __init__(
        self,
        encoder_channels,
        num_cls=21,
        use_batchnorm=True,
        out_channels=512,
        p_dropout=0.2,
    ):
        super().__init__()

        self.psp = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = nn.Conv2d(encoder_channels[-1] * 2, out_channels, (1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.cls_conv = nn.Conv2d(out_channels, num_cls, (1, 1))
        self.p_dropout = p_dropout

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = F.dropout(x, p=self.p_dropout, training=self.get_do_status())
        x=self.cls_conv(x)
        return x


def _pspnet_mobilenetv3(
    backbone: mobilenetv3.MobileNetV3,
    num_classes: int,
    aux: Optional[bool],
    classification_head_loss_features =False,
    deeplabplus=False,
) -> PSPNet:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    # if aux:
    #     return_layers[str(aux_pos)] = "aux"
    # backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None

    classifier = PSPDecoder(out_inplanes, num_classes)
    return PSPNet(backbone, classifier, aux_classifier, process_idx={"out":-1, "aux":-4, "low_level":-4}, classifier_loss_features=classification_head_loss_features)


def _pspnet_resnet(
    backbone: resnet.ResNet,
    num_classes: int,
    aux: Optional[bool],
    classification_head_loss_features: Optional[bool],
    in_channels=(2048,1024,512,256)
) -> PSPNet:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    # backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(in_channels[1], num_classes) if aux else None
    classifier = PSPDecoder(in_channels[0], num_classes)
    return PSPNet(backbone, classifier, aux_classifier,process_idx={"out":-1, "aux":-2, "low_level": -4}, classifier_loss_features=classification_head_loss_features)


def pspnet_resnet34(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    classification_head_loss_features: bool = True,
    deeplabplus = False,
) -> PSPNet:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet34(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, False, False])
    model = _pspnet_resnet(backbone, num_classes, aux_loss,classification_head_loss_features,deeplabplus,[512,256,128,64])

    return model


def pspnet_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    classification_head_loss_features: bool = True,
    deeplabplus = False,
) -> PSPNet:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet34(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, False, False])
    model = _pspnet_resnet(backbone, num_classes, aux_loss,classification_head_loss_features,deeplabplus,[512,256,128,64])

    return model

def pspnet_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    classification_head_loss_features: bool = True,
    deeplabplus = False,
) -> PSPNet:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet34(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, False, False])
    model = _pspnet_resnet(backbone, num_classes, aux_loss,classification_head_loss_features,deeplabplus,[512,256,128,64])

    return model


def pspnet_mobilenet_v3_large(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    classification_head_loss_features: bool = True,
    deeplabplus = False,
) -> PSPNet:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet34(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, False, False])
    model = _pspnet_mobilenetv3(backbone, num_classes, aux_loss,classification_head_loss_features,deeplabplus,[512,256,128,64])

    return model