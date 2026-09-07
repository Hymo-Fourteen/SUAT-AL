'''
ResNet in PyTorch.

'''

import torch.nn as nn
import torch.nn.functional as F


from joda_al.Pytorch_Extentions.activations import PenalizedTanh
from joda_al.models.model_lib import LossLearningMixin


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm_layer=nn.BatchNorm2d, stride=1,act="ReLu"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion * planes)
                norm_layer(self.expansion * planes)
            )
        if act == "TanHP":
            self.activation = PenalizedTanh()
        else:
            self.activation = F.relu

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class BasicBlockDropout(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm_layer=nn.BatchNorm2d, stride=1,act="ReLu"):
        super(BasicBlockDropout, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )
        self.mc_dropout_active = False
        if act == "TanHP":
            self.activation = PenalizedTanh()
        else:
            self.activation = F.relu

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def forward(self, x):
        out = F.dropout(self.activation(self.bn1(self.conv1(x))), p=0.05, training=self.get_do_status())
        out = F.dropout(self.bn2(self.conv2(out)), p=0.05, training=self.get_do_status())
        out += self.shortcut(x)
        out = self.activation(out)
        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm_layer=nn.BatchNorm2d, stride=1, act="ReLu"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.bn3 = norm_layer(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion * planes)
                norm_layer(self.expansion * planes)
            )
        self.mc_dropout_active = False
        if act == "TanHP":
            self.activation = PenalizedTanh()
        else:
            self.activation = F.relu

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module, LossLearningMixin):
    _num_channels = [64, 128, 256, 512]
    def __init__(self, block, num_blocks, num_classes=10, softmax=False,act="ReLu",norm_layer_type="bn",use_during_training=True,first_cov=3,max_pool=False):
        super(ResNet, self).__init__()
        if norm_layer_type == "bn":
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError()
        self.use_during_training = use_during_training
        self.in_planes = 64
        self.embDim = 512
        self.conv1 = nn.Conv2d(3, 64, kernel_size=first_cov, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], norm_layer=norm_layer, stride=1,act=act)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm_layer=norm_layer, stride=2,act=act)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm_layer=norm_layer, stride=2,act=act)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm_layer=norm_layer, stride=2,act=act)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        if act == "TanHP":
            self.activation = PenalizedTanh()
        else:
            self.activation = F.relu

        self.use_softmax = softmax
        # self.linear2 = nn.Linear(1000, num_classes)
        self.mc_dropout_active = False

    def get_loss_learning_features(self, size):
        pass

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        elif self.use_during_training:
            return self.training
        else:
            return False

    def set_dropout_status(self, status):
        self.mc_dropout_active = status
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for k, mod in layer._modules.items():
                mod.mc_dropout_active = status

    def _make_layer(self, block, planes, num_blocks, norm_layer, stride, act="ReLu"):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, norm_layer, stride, act))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        if self.max_pool:  # maxpool missing in contrast to reference from torch vision
            out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        # out = F.avg_pool2d(out4, 4) TA-VAAL paper
        out = self.avgpool(
            out4)  # torch vision: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        outf = out.view(out.size(0), -1)
        out = self.linear(outf)
        if self.use_softmax:
            out = F.softmax(out, -1)
        return out, outf, [out1, out2, out3, out4]

    def get_embedding_dim(self):
        return self.embDim


class ResNetDropout(ResNet):

    def __init__(self, block, num_blocks, num_classes=10, softmax=False, act="ReLu", norm_layer_type="bn",
                 do_ratio=0.05,use_during_training=True,first_cov=3):
        super().__init__(block, num_blocks, num_classes, softmax, act, norm_layer_type,use_during_training,first_cov)
        self.do_ratio = do_ratio

    def forward(self, x):
        # maxpool missing in contrast to reference from torch vision
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        # out = F.avg_pool2d(out4, 4) TA-VAAL paper
        out = self.avgpool(
            out4)  # torch vision: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        outf = out.view(out.size(0), -1)
        outf = F.dropout(outf, p=self.do_ratio, training=self.get_do_status())
        out = self.linear(outf)
        if self.use_softmax:
            out = F.softmax(out, -1)
        return out, outf, [out1, out2, out3, out4]

class ResNetCls(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, softmax=False, act="ReLu", norm_layer_type="bn",
                 do_ratio=0.05, use_during_training=True, first_cov=3):
        super().__init__(block, num_blocks, num_classes, softmax, act, norm_layer_type, use_during_training, first_cov)
        self.do_ratio = do_ratio
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, num_classes)


    def forward(self, x):
        out, outf, [out1, out2, out3, out4] = super(ResNetCls, self).forward(x)
        outf = F.dropout(outf, p=self.do_ratio, training=self.get_do_status())
        out = self.linear(outf)
        out = self.activation(out)
        out = F.dropout(out, p=self.do_ratio, training=self.get_do_status())
        out = self.linear2(out)
        out = self.activation(out)
        out = F.dropout(out, p=self.do_ratio, training=self.get_do_status())
        out = self.linear3(out)
        if self.use_softmax:
            out = F.softmax(out, -1)
        return out, outf, [out1, out2, out3, out4]


def ResNet18(num_classes=10, use_softmax=False, act="ReLu", norm_layer_type="bn",do_ratio=0.05,use_during_training=True,first_cov=3,deep_dropout=False):
    if do_ratio > 0.0:
        if not deep_dropout:
            block = BasicBlock
        else:
            block = BasicBlockDropout
        model = ResNetDropout(block, [2, 2, 2, 2], num_classes, use_softmax, act=act,
                                 norm_layer_type=norm_layer_type, do_ratio=do_ratio,
                                 use_during_training=use_during_training,first_cov=first_cov)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_softmax, act=act, norm_layer_type=norm_layer_type,first_cov=first_cov)

    return model


def ResNet18Cls(num_classes=10, use_softmax=False, act="ReLu", norm_layer_type="bn",do_ratio=0.05,use_during_training=True,first_cov=3,deep_dropout=False):
    if not deep_dropout:
        block = BasicBlock
    else:
        block = BasicBlockDropout
    model = ResNetDropout(block, [2, 2, 2, 2], num_classes, use_softmax, act=act,
                             norm_layer_type=norm_layer_type, do_ratio=do_ratio,
                             use_during_training=use_during_training,first_cov=first_cov)
    return model


def ResNet34(num_classes=10, use_softmax=False, act="ReLu", norm_layer_type="bn",do_ratio=0.05,use_during_training=True,first_cov=3,deep_dropout=False):
    if do_ratio > 0.0:
        if not deep_dropout:
            block = BasicBlock
        else:
            block = BasicBlockDropout

        model = ResNetDropout(block, [3, 4, 6, 3], num_classes, use_softmax, act=act,
                              norm_layer_type=norm_layer_type, do_ratio=do_ratio,
                              use_during_training=use_during_training, first_cov=first_cov)
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, use_softmax, act=act, norm_layer_type=norm_layer_type,first_cov=first_cov)
    return model


def ResNet50(num_classes=10, use_softmax=False, act="ReLu", norm_layer_type="bn",do_ratio=0.05,use_during_training=True,first_cov=3):
    """
    first_cov = 3 -> Benchmark =7 Real versions
    """
    if do_ratio >0.0:
        model = ResNetDropout(Bottleneck, [3, 4, 6, 3], num_classes, use_softmax, act=act,
                              norm_layer_type=norm_layer_type, do_ratio=do_ratio,
                              use_during_training=use_during_training,first_cov=first_cov)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, use_softmax, act=act, norm_layer_type=norm_layer_type,first_cov=first_cov)

    model._num_channels= [256, 512, 1024, 2048]
    model.embDim=2048
    return model

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


