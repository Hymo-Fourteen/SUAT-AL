import torch
from torch import optim

from joda_al.Pytorch_Extentions.scheduler import get_scheduler

from joda_al.utils.method_utils import is_loss_learning
from joda_al.models.SemanticSegmentation.PSPNet import pspnet_mobilenet_v3_large
from joda_al.models.SemanticSegmentation.deeplabV3 import (
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet34,
)
from joda_al.models.query_models.query_models import GenericLossNet


# TODO either here or seperate file add the loading from src/models/__init__
def setup_seg_model(method, device, dataset_config, training_config, cls_counts=None):
    with torch.cuda.device(device):

        if training_config["model"] == "deeplabv3_resnet50":
            task_model=deeplabv3_resnet50(num_classes=dataset_config["num_classes"],pretrained=training_config["pretrained"],aux_loss=training_config["aux_loss"],pretrained_backbone=training_config["pretrained_backbone"],classification_head_loss_features=training_config["head_loss_features"])


            num_channels = [256, 512, 1024, 2048]
            num_channels2 = [256, 512, 1024, 2048, 256, 256, 256, 256, 256, 256]
        elif training_config["model"] == "deeplabv3_resnet34":
            task_model=deeplabv3_resnet34(num_classes=dataset_config["num_classes"],pretrained=training_config["pretrained"],pretrained_backbone=training_config["pretrained_backbone"],classification_head_loss_features=training_config["head_loss_features"],use_npn=use_npn_head,**npn_arg)


            num_channels = [64, 128, 256, 512]
            num_channels2 = [64, 128, 256, 512, 256, 256, 256, 256, 256, 256]
        elif training_config["model"] == "deeplabv3_mobilenet_v3":
            task_model=deeplabv3_mobilenet_v3_large(num_classes=dataset_config["num_classes"],pretrained=training_config["pretrained"],pretrained_backbone=training_config["pretrained_backbone"],classification_head_loss_features=training_config["head_loss_features"])


            num_channels = [40, 80, 160, 960]
            num_channels2 = [40, 80, 160, 960, 256, 256, 256, 256, 256, 256]
        elif training_config["model"] == "deeplabv3plus_resnet50":
            task_model=deeplabv3_resnet50(num_classes=dataset_config["num_classes"],pretrained=training_config["pretrained"],pretrained_backbone=training_config["pretrained_backbone"],classification_head_loss_features=training_config["head_loss_features"],deeplabplus=True)


            num_channels = [256, 512, 1024, 2048]
            num_channels2 = [256, 512, 1024, 2048, 256, 256, 256, 256, 256, 256]
        elif training_config["model"] == "deeplabv3plus_resnet34":
            task_model=deeplabv3_resnet34(num_classes=dataset_config["num_classes"],pretrained=training_config["pretrained"],pretrained_backbone=training_config["pretrained_backbone"],classification_head_loss_features=training_config["head_loss_features"],deeplabplus=True)
            
            num_channels = [64, 128, 256, 512]
            num_channels2 = [64, 128, 256, 512, 256, 256, 256, 256, 256, 256]
        elif training_config["model"] == "deeplabv3plus_resnet34M":
            task_model = deeplabv3_resnet34(
                num_classes=dataset_config["num_classes"],
                pretrained=training_config["pretrained"],
                pretrained_backbone=training_config["pretrained_backbone"],
                classification_head_loss_features=training_config["head_loss_features"],
                deeplabplus=True,
                skip_max_pool=True,
            )
            # num_channels = [64, 128, 256, 512]
            # num_channels2 = [64, 128, 256, 512, 256, 256, 256, 256, 256]
        elif training_config["model"] == "deeplabv3plus_mobilenet_v3":
            task_model = deeplabv3_mobilenet_v3_large(
                num_classes=dataset_config["num_classes"],
                pretrained=training_config["pretrained"],
                pretrained_backbone=training_config["pretrained_backbone"],
                classification_head_loss_features=training_config["head_loss_features"],
                deeplabplus=True,
            )
            num_channels = [40, 80, 160, 960]
            # num_channels2 = [40, 80, 160, 960, 256,256,256,256,256]
        elif training_config["model"] == "psp_mobilenet_v3":
            task_model = pspnet_mobilenet_v3_large(num_classes=dataset_config["num_classes"],pretrained=training_config["pretrained"],pretrained_backbone=training_config["pretrained_backbone"],classification_head_loss_features=training_config["head_loss_features"],deeplabplus=True)
            
            # num_channels = task_model.encorder.num_channels
            # num_channels2 = [40, 80, 160, 960, 256,256,256,256,256]
        else:
            raise NotImplementedError()
        models = {"task": task_model.cuda()}
        if is_loss_learning(method):
            num_channels2 = num_channels + [256, 256, 256, 256, 256, 256]
            feature_sizes = dataset_config["feature_sizes"][training_config["model"]]
            # loss_module = LossNet(feature_sizes=[16,8,4,2], num_channels=[128,128,256,512]).cuda()
            if training_config["head_loss_features"]:
                loss_module = GenericLossNet(
                    feature_sizes + [feature_sizes[-1]] * 6, num_channels2, 128
                )
            else:
                if num_channels is None:
                    num_channels = task_model.backbone.num_channels
                loss_module = GenericLossNet(feature_sizes[-4:], num_channels, 128)
            models["module"] = loss_module.cuda()
        # summary(models["task"], input_size=(3, 300, 480))

        # print(models)
        return models


def get_optimizer_scheduler_semseg(model, training_config):
    if training_config["optimizer"] == "SGD":
        params=[
        {'params': model.backbone.parameters(), 'lr': training_config.get("lr_backbone", training_config["lr"])},
        {'params': model.classifier.parameters(), 'lr': training_config["lr"]},
        ]
        if model.aux_classifier is not None:
            params.append({'params': model.aux_classifier.parameters(), 'lr': training_config["lr"]})
        optimizer = optim.SGD(params=params, lr=training_config["lr"], momentum=training_config["momentum"], weight_decay=training_config["wdcay"])
        scheduler = get_scheduler(training_config,optimizer)
        return optimizer, scheduler
    elif training_config["optimizer"]== "Adam":
        optimizer=optim.Adam(model.parameters(), lr=training_config["lr"], weight_decay=training_config["wdcay"])
        return optimizer, None
    elif training_config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=training_config["lr"], weight_decay=training_config["wdcay"],momentum=training_config["momentum"])
        return optimizer, None
    else:
        raise NotImplementedError()
