from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from joda_al.models.model_lib import McDropoutModelMixin


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module,McDropoutModelMixin):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
       #     nn.Dropout(0.5), moved dropout to forward
        )
        self.mc_dropout_active = False
        self.dropout_probability = 0.1

    def get_do_status(self):
        if self.mc_dropout_active:
            return self.mc_dropout_active
        else:
            return self.training

    def set_dropout_status(self, status):
        self.mc_dropout_active = status

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        # print("r")
        # for r in _res:
        #     print(r.shape)
        # r:
        # torch.Size([32, 256, 19, 30])
        # torch.Size([32, 256, 19, 30])
        # torch.Size([32, 256, 19, 30])
        # torch.Size([32, 256, 19, 30])
        # torch.Size([32, 256, 19, 30])
        res = torch.cat(_res, dim=1)
        x = self.project(res)
        x = F.dropout(x, p=self.dropout_probability, training=self.get_do_status())
        return x, _res