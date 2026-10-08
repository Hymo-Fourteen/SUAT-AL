"""RxRx1 数据集的 JODA 适配器（**薄层**）。

为什么只有这一层放在 JODA 里
---------------------------
JODA 的数据集工厂要求可调用对象能在 `joda_al.data_loaders.classification` 里被
import 到（见 `load_classification_dataset.py` 的 `dataset_factory`），所以至少要有
一个模块落在这里。实验真实的数据/划分/变换代码全部放在仓库根的
`exp/rxrx1/common/`，本文件只做三件事：

1. 把仓库根加入 ``sys.path``（JODA 以 ``PYTHONPATH=src`` 从 `JODA/` 目录启动，
   仓库根默认不可见）；
2. 注入只有在 JODA 内部才能构造的 `DatasetScenarioConverter`；
3. 转发调用，并把两个变换实例暴露给 ``trans_map``。

这样上游 JODA 更新时的冲突面最小，实验逻辑也不受 JODA 重构影响。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from joda_al.data_loaders.DatasetScenarioConverter import DatasetScenarioConverter

# JODA/src/joda_al/data_loaders/classification/<this file> -> 仓库根
_REPO_ROOT = Path(__file__).resolve().parents[5]
_COMMON_MODULE = Path("exp") / "rxrx1" / "common" / "rxrx1_dataset.py"


def _ensure_common_importable() -> None:
    module_path = _REPO_ROOT / _COMMON_MODULE
    if not module_path.is_file():
        raise ImportError(
            f"找不到实验共享代码 {module_path}。\n"
            f"RxRx1 loader 依赖仓库根下的 exp/rxrx1/common/，请确认本仓库结构完整，"
            f"或把 JODA/ 作为子目录放回仓库中。"
        )
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


_ensure_common_importable()

# 供 `load_classification_dataset.py` 的两个 `trans_map` 使用。
# 参见 exp/rxrx1/common/transforms.py：训练变换必须 tensor-safe 且不含 ToTensor，
# 因为它会叠在 eval 变换（已把池转成标准化张量）之上。
from exp.rxrx1.common.transforms import (  # noqa: E402
    EVAL_TRANSFORM as RXRX1_EVAL_TRANSFORM,
    TRAIN_TRANSFORM as RXRX1_TRAIN_TRANSFORM,
)

__all__ = ["load_rxrx1", "RXRX1_TRAIN_TRANSFORM", "RXRX1_EVAL_TRANSFORM"]


def load_rxrx1(dataset: Dict, config: Any, order: int = 1):
    """`dataset_factory` 中的 ``"rxrx1"`` 入口，签名与其它 loader 保持一致。"""
    from exp.rxrx1.common.rxrx1_dataset import load_rxrx1 as _impl

    return _impl(dataset, config, order, DatasetScenarioConverter())
