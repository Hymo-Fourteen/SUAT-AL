import os
from urllib.request import URLopener

import torch
import torch.nn as nn
import torch.nn.functional as F

feature_sizes = [64, 128, 256, 512, 256, 128, 64]


class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        return_logits=False,
        width_mult=1,
    ):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.feature_sizes = [int(size * width_mult) for size in feature_sizes]

        self.conv1_1 = nn.Conv2d(
            in_channels, self.feature_sizes[0], 3, padding=1, bias=False
        )
        self.conv1_1_bn = nn.BatchNorm2d(self.feature_sizes[0])
        self.conv1_2 = nn.Conv2d(
            self.feature_sizes[0], self.feature_sizes[0], 3, padding=1, bias=False
        )
        self.conv1_2_bn = nn.BatchNorm2d(self.feature_sizes[0])

        self.conv2_1 = nn.Conv2d(
            self.feature_sizes[0], self.feature_sizes[1], 3, padding=1, bias=False
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.feature_sizes[1])
        self.conv2_2 = nn.Conv2d(
            self.feature_sizes[1], self.feature_sizes[1], 3, padding=1, bias=False
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.feature_sizes[1])

        self.conv3_1 = nn.Conv2d(
            self.feature_sizes[1], self.feature_sizes[2], 3, padding=1, bias=False
        )
        self.conv3_1_bn = nn.BatchNorm2d(self.feature_sizes[2])
        self.conv3_2 = nn.Conv2d(
            self.feature_sizes[2], self.feature_sizes[2], 3, padding=1, bias=False
        )
        self.conv3_2_bn = nn.BatchNorm2d(self.feature_sizes[2])
        self.conv3_3 = nn.Conv2d(
            self.feature_sizes[2], self.feature_sizes[2], 3, padding=1, bias=False
        )
        self.conv3_3_bn = nn.BatchNorm2d(self.feature_sizes[2])

        self.conv4_1 = nn.Conv2d(
            self.feature_sizes[2], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv4_1_bn = nn.BatchNorm2d(self.feature_sizes[3])
        self.conv4_2 = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv4_2_bn = nn.BatchNorm2d(self.feature_sizes[3])
        self.conv4_3 = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv4_3_bn = nn.BatchNorm2d(self.feature_sizes[3])

        self.conv5_1 = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv5_1_bn = nn.BatchNorm2d(self.feature_sizes[3])
        self.conv5_2 = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv5_2_bn = nn.BatchNorm2d(self.feature_sizes[3])
        self.conv5_3 = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv5_3_bn = nn.BatchNorm2d(self.feature_sizes[3])

        self.conv5_3_D = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv5_3_D_bn = nn.BatchNorm2d(self.feature_sizes[3])
        self.conv5_2_D = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv5_2_D_bn = nn.BatchNorm2d(self.feature_sizes[3])
        self.conv5_1_D = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv5_1_D_bn = nn.BatchNorm2d(self.feature_sizes[3])

        self.conv4_3_D = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv4_3_D_bn = nn.BatchNorm2d(self.feature_sizes[3])
        self.conv4_2_D = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[3], 3, padding=1, bias=False
        )
        self.conv4_2_D_bn = nn.BatchNorm2d(self.feature_sizes[3])
        self.conv4_1_D = nn.Conv2d(
            self.feature_sizes[3], self.feature_sizes[4], 3, padding=1, bias=False
        )
        self.conv4_1_D_bn = nn.BatchNorm2d(self.feature_sizes[4])

        self.conv3_3_D = nn.Conv2d(
            self.feature_sizes[4], self.feature_sizes[4], 3, padding=1, bias=False
        )
        self.conv3_3_D_bn = nn.BatchNorm2d(self.feature_sizes[4])
        self.conv3_2_D = nn.Conv2d(
            self.feature_sizes[4], self.feature_sizes[4], 3, padding=1, bias=False
        )
        self.conv3_2_D_bn = nn.BatchNorm2d(self.feature_sizes[4])
        self.conv3_1_D = nn.Conv2d(
            self.feature_sizes[4], self.feature_sizes[5], 3, padding=1, bias=False
        )
        self.conv3_1_D_bn = nn.BatchNorm2d(self.feature_sizes[5])

        self.conv2_2_D = nn.Conv2d(
            self.feature_sizes[5], self.feature_sizes[5], 3, padding=1, bias=False
        )
        self.conv2_2_D_bn = nn.BatchNorm2d(self.feature_sizes[5])
        self.conv2_1_D = nn.Conv2d(
            self.feature_sizes[5], self.feature_sizes[6], 3, padding=1, bias=False
        )
        self.conv2_1_D_bn = nn.BatchNorm2d(self.feature_sizes[6])

        self.conv1_2_D = nn.Conv2d(
            self.feature_sizes[6], self.feature_sizes[6], 3, padding=1, bias=False
        )
        self.conv1_2_D_bn = nn.BatchNorm2d(self.feature_sizes[6])
        self.conv1_1_D = nn.Conv2d(
            self.feature_sizes[6], num_classes, 3, padding=1, bias=False
        )

        self.apply(self.weight_init)

        self.return_logits = return_logits

    def forward(self, x: torch.Tensor):
        # Encoder block 1
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x, mask1 = self.pool(x)

        # Encoder block 2
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(F.relu(self.conv2_2(x))))
        x, mask2 = self.pool(x)

        # Encoder block 3
        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x = F.relu(self.conv3_3_bn(self.conv3_3(x)))
        x, mask3 = self.pool(x)

        # Encoder block 4
        x = F.relu(self.conv4_1_bn(self.conv4_1(x)))
        x = F.relu(self.conv4_2_bn(self.conv4_2(x)))
        x = F.relu(self.conv4_3_bn(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = F.relu(self.conv5_1_bn(self.conv5_1(x)))
        x = F.relu(self.conv5_2_bn(self.conv5_2(x)))
        x = F.relu(self.conv5_3_bn(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = F.relu(self.conv5_3_D_bn(self.conv5_3_D(x)))
        x = F.relu(self.conv5_2_D_bn(self.conv5_2_D(x)))
        x = F.relu(self.conv5_1_D_bn(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = F.relu(self.conv4_3_D_bn(self.conv4_3_D(x)))
        x = F.relu(self.conv4_2_D_bn(self.conv4_2_D(x)))
        x = F.relu(self.conv4_1_D_bn(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = F.relu(self.conv3_3_D_bn(self.conv3_3_D(x)))
        x = F.relu(self.conv3_2_D_bn(self.conv3_2_D(x)))
        x = F.relu(self.conv3_1_D_bn(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = F.relu(self.conv2_2_D_bn(self.conv2_2_D(x)))
        x = F.relu(self.conv2_1_D_bn(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = F.relu(self.conv1_2_D_bn(self.conv1_2_D(x)))

        x = self.conv1_1_D(x)

        if not self.return_logits:
            x = F.log_softmax(x)

        return x


def segnet(**kwargs):
    net = SegNet(**kwargs)

    return net
