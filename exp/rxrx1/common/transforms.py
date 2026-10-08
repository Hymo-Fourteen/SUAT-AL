"""RxRx1 的输入变换。

为什么不用 ImageNet 归一化
--------------------------
RxRx1 是荧光显微图像，绝大部分像素接近全黑（全图均值约 0.02–0.06），且不同
板/孔的曝光差异很大。改用 ImageNet 的固定 mean/std 会让 energy 分数出现
**假性可分**（见 `protocol.md` §3.3 与 notes.md 的记录）。因此这里采用
**逐图实例标准化**：对每张图、每个通道单独减均值除标准差。

为什么训练变换里不能有 ``ToTensor``
-----------------------------------
`load_classification_dataset` 对非白名单数据集会用 **eval 变换包裹整个池**
（这是必需的：`get_unlabeled_pool_loader` 直接吃 `self.train_pool`，能源评分必须
拿到 tensor），随后 `get_train_loader` 再把**训练变换叠在已标注子集上**。
所以训练变换拿到的是已经标准化过的 tensor，必须是 tensor-safe 且**不能**再做
一次 ``ToTensor``。90° 旋转与水平翻转都是纯像素置换，先标准化再增强与
先增强再标准化等价，因此这个顺序在语义上是安全的。

``Rotate90`` 只做 90° 的整数倍旋转（用 ``torch.rot90`` / ``PIL.rotate``），
不做插值，因此不会引入填充值污染标准化后的张量。
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import Compose


class InstanceStandardize:
    """PIL / ndarray / tensor -> ``float32`` 的 CHW 张量，并做逐通道实例标准化。

    把它当成「安全的 ToTensor + 归一化」二合一：输入端接受图像或张量，
    输出端保证是 ``(3, H, W)`` 的 ``float32`` 张量，均值约 0、方差约 1。

    这样 far-OOD（F2 噪声）路径可以直接喂张量，不需要让 ``ToTensor`` 去猜
    「已经是张量」时该怎么处理。
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)

    def __call__(self, image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(image))
            if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):  # HWC -> CHW
                tensor = tensor.permute(2, 0, 1)
        else:  # PIL.Image
            tensor = TF.pil_to_tensor(image)  # uint8, CHW

        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32).div(255.0)
        elif tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        if tensor.ndim != 3:
            raise ValueError(f"期望 (C, H, W) 张量，得到 shape={tuple(tensor.shape)}")

        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), unbiased=False, keepdim=True)
        return (tensor - mean) / (std + self.eps)


class Rotate90:
    """随机旋转 0 / 90 / 180 / 270 度（无插值，无填充）。

    既接受 PIL 图像也接受张量，便于单独测试与复用。
    """

    def __call__(self, image: Any) -> Any:
        quarters = random.randint(0, 3)
        if quarters == 0:
            return image
        if isinstance(image, torch.Tensor):
            return torch.rot90(image, quarters, dims=(-2, -1)).contiguous()
        if isinstance(image, np.ndarray):
            return np.rot90(image, quarters, axes=(0, 1)).copy()
        return image.rotate(90 * quarters, expand=True, resample=False)  # PIL


def rxrx1_eval_transform() -> Compose:
    """确定性的评测/池变换：PIL -> 标准化张量。

    由 `load_classification_dataset` 施加到池、验证集与测试集上。
    """
    return Compose([InstanceStandardize()])


def rxrx1_train_transform() -> Compose:
    """训练增强：90° 旋转 + 水平翻转。

    **不含** ``ToTensor``/标准化——输入已经是标准化后的张量，见模块 docstring。
    """
    from torchvision.transforms import RandomHorizontalFlip

    return Compose([Rotate90(), RandomHorizontalFlip()])


# 与 JODA 的 `trans_map` 用法保持一致：模块级复用同一实例即可
# （这些变换在 `__call__` 里使用全局 RNG，本身无状态）。
EVAL_TRANSFORM = rxrx1_eval_transform()
TRAIN_TRANSFORM = rxrx1_train_transform()
