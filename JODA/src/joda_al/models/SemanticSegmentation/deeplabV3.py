from collections import OrderedDict
from typing import Optional, Dict

import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

# from torchvision.models import mobilenetv3
from joda_al.models import torchMobilenet as mobilenetv3
from torchvision.utils import _log_api_usage_once

# from torchvision.models import resnet
from joda_al.models import torchResnet as resnet
from torchvision.models.segmentation.fcn import FCNHead

from joda_al.models.SemanticSegmentation.aspp import ASPP
from joda_al.models.model_lib import (
    McDropoutModelMixin,
    McDropoutMixin,
    LossLearningMixin,
)


class _SimpleSegmentationModel(nn.Module, McDropoutModelMixin):
    __constants__ = ["aux_classifier"]

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None,
        process_idx: Optional[Dict] = None,
        classifier_loss_features: Optional[bool] = False,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.process_idx = process_idx
        self.mc_dropout_active = False
        self.classifier_loss_features = classifier_loss_features

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def set_dropout_status(self, status):
        self.mc_dropout_active = status
        self.classifier.set_dropout_status(status)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        _, flat_features, features = self.backbone(x)

        backbone_features = {k: features[v] for k, v in self.process_idx.items()}
        result = OrderedDict()
        x = features[-1]
        # print(backbone_features)
        x, cls_loss_features = self.classifier(backbone_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x
        flat_features = torch.flatten(features[-1], 1)
        if self.aux_classifier is not None:
            x = backbone_features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x
            flat_features = torch.flatten(backbone_features["aux"], 1)
        if self.classifier_loss_features:
            features += cls_loss_features

        return result, flat_features, features


def _load_weights(
    arch: str, model: nn.Module, model_url: Optional[str], progress: bool
) -> None:
    if model_url is None:
        raise ValueError(f"No checkpoint is available for {arch}")
    state_dict = load_state_dict_from_url(
        model_url, model_dir="../Model_Weights", progress=progress
    )
    # Ignore last layer
    last_layer_keys = [
        "classifier.4.bias",
        "classifier.4.weight",
        "aux_classifier.4.bias",
        "aux_classifier.4.weight",
    ]
    for key in last_layer_keys:
        del state_dict[key]
    last_layer_keys += ["backbone.fc.weight", "backbone.fc.bias"]
    report = model.load_state_dict(state_dict, strict=False)
    assert sorted(report.missing_keys) == sorted(last_layer_keys)


__all__ = [
    "DeepLabV3",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
]


model_urls = {
    "deeplabv3_resnet50_coco": "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
    "deeplabv3_resnet101_coco": "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
    "deeplabv3_mobilenet_v3_large_coco": "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
}


class DeepLabV3(_SimpleSegmentationModel, LossLearningMixin):
    _num_channels = [256, 256, 256, 256, 256]

    def get_loss_learning_features(self, size):
        return list(self.backbone.get_loss_learning_features(size)[-1]) * 5

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


class DeepLabClassifier(nn.Sequential):
    def __init__(self, in_channels, hidden_size, num_classes):
        modules = [
            nn.Conv2d(in_channels, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, num_classes, 1),
        ]
        super().__init__(*modules)
        self.dropout_probability = 0.1
        self.mc_dropout_active = False

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def forward(self, feature):
        x = self[0](feature)
        x = self[1](x)
        x = self[2](x)
        x = F.dropout(x, p=self.dropout_probability, training=self.get_do_status())
        x = self[3](x)
        return x


class DeepLabHeadV3Plus(nn.Module, McDropoutModelMixin):
    def __init__(
        self, in_channels, low_level_channels, num_classes, aspp_dilate=[1, 12, 24, 36]
    ):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        # self.classifier = nn.Sequential(
        #     nn.Conv2d(304, 256, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, num_classes, 1)
        # )
        self.classifier = DeepLabClassifier(304, 256, num_classes)
        self._init_weight()

        self.mc_dropout_active = False

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def set_dropout_status(self, status):
        self.mc_dropout_active = status
        # self.ASPP.mc_dropout_active = status
        self.aspp.mc_dropout_active = status
        self.classifier.mc_dropout_active = status

    def forward(self, feature):
        low_level_feature = self.project(feature["low_level"])
        output_feature, loss_features = self.aspp(feature["out"])
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return (
            self.classifier(torch.cat([low_level_feature, output_feature], dim=1)),
            loss_features,
        )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Sequential, McDropoutModelMixin):
    def __init__(
        self, in_channels: int, num_classes: int, aspp_dilate=[12, 24, 36]
    ) -> None:
        self.layers = [
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        ]
        super().__init__(*self.layers)
        # super().__init__()
        # self.aspp = ASPP(in_channels, [12, 24, 36])
        # self.conv1=nn.Conv2d(256, 256, 3, padding=1, bias=False)
        # self.bn1= nn.BatchNorm2d(256)
        # self.relu= nn.ReLU()
        # self.conv2=nn.Conv2d(256, num_classes, 1)

        self.mc_dropout_active = False

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def set_dropout_status(self, status):
        self.mc_dropout_active = status
        # self.ASPP.mc_dropout_active = status
        self.layers[0].mc_dropout_active = status

    def forward(self, x: torch.Tensor):
        x, features = self.layers[0](x["out"])
        for layer in self.layers[1:]:
            x = layer(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        return x, features


class NpnDeepLabHeadV3Plus(DeepLabHeadV3Plus):
    def __init__(
        self,
        in_channels,
        low_level_channels,
        num_classes,
        aspp_dilate=[12, 24, 36],
        **kwargs,
    ):
        super(NpnDeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, 1),
            NPNuncertaintyLayer(32, num_classes, layer="conv", **kwargs),
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature["low_level"])
        output_feature, loss_features = self.aspp(feature["out"])
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return (
            self.classifier(torch.cat([low_level_feature, output_feature], dim=1)),
            loss_features,
        )


class NpnDeepLabHead(nn.Sequential, McDropoutMixin):
    def __init__(
        self, in_channels: int, num_classes: int, aspp_dilate=[12, 24, 36], **kwargs
    ) -> None:
        self.layers = [
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 32, 1),
            NPNuncertaintyLayer(32, num_classes, layer="conv", **kwargs),
        ]
        super().__init__(*self.layers)
        # super().__init__()
        # self.aspp = ASPP(in_channels, [12, 24, 36])
        # self.conv1=nn.Conv2d(256, 256, 3, padding=1, bias=False)
        # self.bn1= nn.BatchNorm2d(256)
        # self.relu= nn.ReLU()
        # self.conv2=nn.Conv2d(256, num_classes, 1)

        self.mc_dropout_active = False

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def set_dropout_status(self, status):
        self.mc_dropout_active = status
        # self.ASPP.mc_dropout_active = status
        self.layers[0].mc_dropout_active = status

    def forward(self, x: torch.Tensor):
        x, features = self.layers[0](x["out"])
        for layer in self.layers[1:]:
            x = layer(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        return x, features


def _deeplabv3_resnet(
    backbone: resnet.ResNet,
    num_classes: int,
    aux: Optional[bool],
    classification_head_loss_features: Optional[bool],
    deeplabplus=False,
    in_channels=(2048, 1024, 512, 256),
    use_npn=False,
    **kwargs,
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    # backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(in_channels[1], num_classes) if aux else None
    classifier = head_factory(
        in_channels[0], in_channels[3], num_classes, deeplabplus, use_npn, **kwargs
    )
    return DeepLabV3(
        backbone,
        classifier,
        aux_classifier,
        process_idx={"out": -1, "aux": -2, "low_level": -4},
        classifier_loss_features=classification_head_loss_features,
    )


def head_factory(
    out_inplanes, aux_inplanes, num_classes, deeplabplus=False, **kwargs
):
    if deeplabplus:
        classifier = DeepLabHeadV3Plus(out_inplanes, aux_inplanes, num_classes)
    else:
        classifier = DeepLabHead(out_inplanes, num_classes)

    return classifier


def _deeplabv3_mobilenetv3(
    backbone: mobilenetv3.MobileNetV3,
    num_classes: int,
    aux: Optional[bool],
    classification_head_loss_features=False,
    deeplabplus=False,
) -> DeepLabV3:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = (
        [0]
        + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)]
        + [len(backbone) - 1]
    )
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    # if aux:
    #     return_layers[str(aux_pos)] = "aux"
    # backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    if deeplabplus:
        classifier = DeepLabHeadV3Plus(out_inplanes, aux_inplanes, num_classes)
    else:
        classifier = DeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(
        backbone,
        classifier,
        aux_classifier,
        process_idx={"out": -1, "aux": -4, "low_level": -4},
        classifier_loss_features=classification_head_loss_features,
    )


def deeplabv3_resnet34(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    classification_head_loss_features: bool = True,
    deeplabplus=False,
    skip_max_pool=False,
    use_npn=False,
    **kwargs,
) -> DeepLabV3:
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

    backbone = resnet.resnet34(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, False, False],
        skip_max_pool=skip_max_pool,
    )
    model = _deeplabv3_resnet(
        backbone,
        num_classes,
        aux_loss,
        classification_head_loss_features,
        deeplabplus,
        [512, 256, 128, 64],
        use_npn,
        **kwargs,
    )

    if pretrained:
        arch = "deeplabv3_resnet34_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def deeplabv3_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    classification_head_loss_features: bool = True,
    deeplabplus=False,
) -> DeepLabV3:
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

    backbone = resnet.resnet50(
        pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True]
    )
    model = _deeplabv3_resnet(
        backbone, num_classes, aux_loss, classification_head_loss_features, deeplabplus
    )

    if pretrained:
        arch = "deeplabv3_resnet50_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def deeplabv3_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    classification_head_loss_features: bool = True,
    deeplabplus=False,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool, optional): If True, include an auxiliary classifier
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet101(
        pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True]
    )
    model = _deeplabv3_resnet(
        backbone, num_classes, aux_loss, classification_head_loss_features, deeplabplus
    )

    if pretrained:
        arch = "deeplabv3_resnet101_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def deeplabv3_mobilenet_v3_large(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
    classification_head_loss_features=False,
    deeplabplus=False,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
        :param classification_head_loss_features:
    """
    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = mobilenetv3.mobilenet_v3_large(
        pretrained=pretrained_backbone, dilated=True
    )
    model = _deeplabv3_mobilenetv3(
        backbone, num_classes, aux_loss, classification_head_loss_features, deeplabplus
    )

    if pretrained:
        arch = "deeplabv3_mobilenet_v3_large_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model
