"""RxRx1 的 JODA 数据集与池构造。

本模块**不 import `joda_al`**：`load_rxrx1` 需要的 `DatasetScenarioConverter`
由适配器（`JODA/src/joda_al/data_loaders/classification/rxrx1_data_loader.py`）
作为参数注入。这样离线工具（划分生成、离线评测、缓存构建）可以在没有 JODA 的
环境里复用同一份数据代码。

标签空间约定
------------
池中的**原始标签**必须直接落在 JODA 期望的最终标签空间上：

* InD 类           -> ``0 .. I-1``
* Discovery 类     -> ``I .. I+D-1``（未提升前即为 ``>= num_classes``）
* far-OOD          -> ``-1``

`apply_open_set_modifications` 之后会把池交给 `AlteredDataset`，其
``indmap``/``nearoodmap`` 恰好是恒等映射，因此这套编号是自洽的
（见 notes.md 关于 ``ood_stats = (label < 0)*2 + (label >= num_class)`` 的推导）。

池的顺序同样有约定：**初始标注集必须是前缀**，之后依次是剩余目标样本与
far-OOD。`label_idx`` 因此就是 ``range(len(initial))``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from exp.rxrx1.common.splits import (
    IMAGE_REL_TEMPLATE,
    load_metadata,
    load_split,
    pool_frame,
    split_indices,
)

DEFAULT_SAMPLE_SIZE = 256
NOISE_KIND = "noise"
FILE_KIND = "file"


class RxRx1ImageListDataset(Dataset):
    """显式样本清单数据集，暴露 JODA 需要的 ``targets`` / ``get_class_counts`` 契约。

    ``records`` 的每一项是 ``(path_or_none, target, kind, cell_type, experiment)``：

    * ``kind == "file"``：``path_or_none`` 是图片绝对路径；``cell_type``/``experiment``
      供离线按细胞系/批次分组统计（这些属性会被 `AlteredDataset`/`TransformDataset`
      自动复制到外层，见 notes.md）。
    * ``kind == "noise"``：far-OOD 的 F2 来源，按索引种子即时生成 ``randn`` 张量，
      不占磁盘也不需要外部数据。种子固定保证同一索引在多次遍历中取值一致
      （能源评分会对池做多轮遍历，样本若随机变化则阈值不稳定）。
    """

    def __init__(
        self,
        records: Sequence[Tuple[Optional[str], int, str, str, str]],
        *,
        image_root: Optional[Path] = None,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        noise_seed: int = 0,
        transform: Any = None,
    ) -> None:
        self.records = list(records)
        self.image_root = str(image_root) if image_root is not None else None
        self.sample_size = int(sample_size)
        self.noise_seed = int(noise_seed)
        self.transform = transform

        # JODA 契约：targets 是平行于索引的标签列表
        self.targets = [int(r[1]) for r in self.records]
        self.kinds = [r[2] for r in self.records]
        self.cell_types = [r[3] for r in self.records]
        self.experiments = [r[4] for r in self.records]
        self.samples = [(r[0], int(r[1])) for r in self.records]
        self.classes = sorted({t for t in self.targets if t >= 0})

    def __len__(self) -> int:
        return len(self.records)

    def _load_noise(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.noise_seed * 1_000_003 + index)
        return torch.randn(3, self.sample_size, self.sample_size, generator=generator)

    def __getitem__(self, index: int):
        path, target, kind, _, _ = self.records[index]
        if kind == NOISE_KIND:
            image: Any = self._load_noise(index)
        else:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def get_class_counts(self, indices: List[int]) -> Dict[int, int]:
        values, counts = np.unique(np.asarray(self.targets)[indices], return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))

    def cell_type_of(self, index: int) -> str:
        return self.cell_types[index]


# --------------------------------------------------------------------------- #
# 记录构造
# --------------------------------------------------------------------------- #
def _rel_paths(frame) -> List[str]:
    return [
        IMAGE_REL_TEMPLATE.format(experiment=e, plate=p, well=w, site=s)
        for e, p, w, s in zip(
            frame["experiment"], frame["plate"], frame["well"], frame["site"]
        )
    ]


def _file_records(frame, rel_paths: Sequence[str], image_root: Path, target_map, label) -> List:
    cell_types = frame["cell_type"].tolist()
    experiments = frame["experiment"].tolist()
    sirna_ids = frame["sirna_id"].tolist()
    return [
        (
            str(image_root / rel),
            int(target_map[sirna_ids[i]]) if label is None else int(label),
            FILE_KIND,
            cell_types[i],
            experiments[i],
        )
        for i, rel in enumerate(rel_paths)
    ]


def _check_images_exist(records: Sequence, limit: int = 5) -> None:
    missing = [r[0] for r in records if r[2] == FILE_KIND and not Path(r[0]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"有 {len(missing)} 张图片不存在，例如: {missing[:limit]}"
        )


def build_pool_and_splits(
    *,
    split: Dict[str, Any],
    image_root: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    verify_files: bool = True,
) -> Dict[str, Any]:
    """按冻结的划分文件构造池 / 验证集 / 测试集记录。

    返回一个 dict，含 ``training_pool`` / ``validation_set`` / ``test_set``
    （均为 ``RxRx1ImageListDataset``，但池的 transform 由 JODA 侧包裹）
    以及用于填 ``dataset_config`` 的元信息。
    """
    ind_orig: List[int] = list(split["class_roles"]["ind_original_class_ids"])
    disc_orig: List[int] = list(split["class_roles"]["discovery_original_class_ids"])
    target_orig = ind_orig + disc_orig
    target_map = {orig: new for new, orig in enumerate(target_orig)}
    initial_per_class = int(split["meta"]["initial_per_class"])
    initial_labeled: Dict[int, List[str]] = {
        int(k): list(v) for k, v in split["initial_labeled"].items()
    }

    meta = load_metadata(image_root)
    pool = pool_frame(meta, target_orig)
    rel_paths = _rel_paths(pool)
    sirna_ids = pool["sirna_id"].tolist()
    pool_cell_types = pool["cell_type"].tolist()
    pool_experiments = pool["experiment"].tolist()

    initial_keys = {
        (cid, rel) for cid, rels in initial_labeled.items() for rel in rels
    }
    is_initial = np.fromiter(
        ((cid, rel) in initial_keys for cid, rel in zip(sirna_ids, rel_paths)),
        dtype=bool,
        count=len(rel_paths),
    )

    # 初始标注集：按 (类别, 路径) 排序，保证与 metadata 行序无关的确定性前缀
    initial_order = sorted(
        range(len(rel_paths)), key=lambda i: (sirna_ids[i], rel_paths[i])
    )
    initial_idx = [i for i in initial_order if is_initial[i]]
    remaining_idx = [i for i in range(len(rel_paths)) if not is_initial[i]]
    if len(initial_idx) != initial_per_class * len(ind_orig):
        raise ValueError(
            f"初始标注集大小为 {len(initial_idx)}，期望 "
            f"{initial_per_class} x {len(ind_orig)} = {initial_per_class * len(ind_orig)}"
        )

    pool_records: List = [
        (
            str(image_root / rel_paths[i]),
            int(target_map[sirna_ids[i]]),
            FILE_KIND,
            pool_cell_types[i],
            pool_experiments[i],
        )
        for i in initial_idx + remaining_idx
    ]
    label_idx = list(range(len(initial_idx)))

    # ---- far-OOD ----
    far_spec = split.get("far_ood")
    far_label = -1
    if far_spec:
        source = far_spec["source"]
        n_far = int(far_spec["n_samples"])
        if source == "noise":
            pool_records.extend(
                (None, far_label, NOISE_KIND, "far_ood", "far_ood") for _ in range(n_far)
            )
        elif source == "dir":
            from exp.rxrx1.common.splits import list_far_ood_images

            far_paths = list_far_ood_images(far_spec)
            pool_records.extend(
                (p, far_label, FILE_KIND, "far_ood", "far_ood") for p in far_paths
            )
        else:
            raise ValueError(f"未知的 far-OOD source: {source!r}")

    # ---- 验证 / 测试：包含全部目标类（InD + Discovery） ----
    # 必须包含 D 类：`reinit_altered_propertysubset_set` 会在每轮提升后按当前
    # ind_classes 重建子集，只有 D 类在集合里，"发现新类"才会体现在 avg_acc 上。
    index = split_indices(meta)
    split_frames = {}
    for name in ("ood_val", "id_test"):
        frame = meta.loc[index[name]]
        frame = frame[frame["sirna_id"].isin(set(target_orig))]
        split_frames[name] = (
            frame,
            _file_records(frame, _rel_paths(frame), image_root, target_map, None),
        )

    if verify_files:
        _check_images_exist(pool_records)
        for _, records in split_frames.values():
            _check_images_exist(records)

    unlabeled_idx = list(range(len(label_idx), len(pool_records)))

    return {
        "pool_records": pool_records,
        "label_idx": label_idx,
        "unlabeled_idx": unlabeled_idx,
        "val_records": split_frames["ood_val"][1],
        "test_records": split_frames["id_test"][1],
        "ind_orig": ind_orig,
        "disc_orig": disc_orig,
        "target_orig": target_orig,
        "class_names": {int(k): v for k, v in split.get("class_names", {}).items()},
        "initial_per_class": initial_per_class,
        "far_spec": far_spec,
        "noise_seed": int((far_spec or {}).get("seed", 0)),
    }


def feature_sizes(sample_size: int = DEFAULT_SAMPLE_SIZE) -> Dict[str, List[int]]:
    """torchvision ResNet 在 256px 输入下 4 个 stage 的**空间尺寸**。

    供 `GenericLossNet`（``AvgPool2d(f_size)``）使用。注意 ``Joda`` 不属于
    `LOSS_MODULE_METHODS`（``['lloss']``），所以该字段在当前主实验中并不会被读取，
    填上只是为了未来对比 ``lloss`` 时不至于缺键。
    """
    size = int(sample_size)
    spatial = [size // 4, size // 8, size // 16, size // 32]
    return {model: list(spatial) for model in ("Resnet18", "Resnet18T", "Resnet34", "Resnet50")}


def load_rxrx1(
    dataset: Dict,
    config: Any,
    order: int = 1,
    scenario_converter: Any = None,
):
    """JODA 数据集工厂入口。

    ``scenario_converter`` 由适配器注入（本模块不依赖 `joda_al`）。
    返回值与 `load_classification_dataset` 期望的 7 元组完全一致。
    """
    del order  # 池的顺序由划分文件唯一决定
    data_config = config.get("data_config", {}) or {}
    split_path = data_config.get("split_file")
    if not split_path:
        raise ValueError(
            "data_config.split_file 未配置；请指向由 "
            "exp/rxrx1/exp1_osal/scripts/make_splits.py 生成的划分文件。"
        )
    split_path = Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"找不到划分文件: {split_path}")
    split = load_split(split_path)

    image_root = Path(data_config.get("image_root") or dataset["path"])
    sample_size = int(data_config.get("image_size", DEFAULT_SAMPLE_SIZE))
    built = build_pool_and_splits(
        split=split,
        image_root=image_root,
        sample_size=sample_size,
        verify_files=bool(data_config.get("verify_files", False)),
    )

    ind_orig = built["ind_orig"]
    disc_orig = built["disc_orig"]
    num_ind, num_disc = len(ind_orig), len(disc_orig)
    total_classes = num_ind + num_disc

    # 这套编号是池标签的硬契约：ind_classes 必须是 0..I-1，
    # 否则 apply_open_set_modifications 里的 near_classes 计算会把 D 类错位。
    expected = list(range(num_ind))
    if list(config["ind_classes"]) != expected:
        raise ValueError(
            f"ind_classes 必须是 {expected[:3]}...{expected[-1:]}（共 {num_ind} 项，"
            f"即 list(range({num_ind}))），实际得到 {config['ind_classes']}。\n"
            f"真实 sirna_id 应写在划分文件里，而不是 ind_classes。"
        )
    if list(config.get("far_classes", [])) != [-1]:
        raise ValueError(
            f"本 loader 的 far-OOD 标签是 -1，请把 far_classes 配成 [-1]，"
            f"实际得到 {config.get('far_classes')}。"
        )

    training_pool = RxRx1ImageListDataset(
        built["pool_records"],
        image_root=image_root,
        sample_size=sample_size,
        noise_seed=built["noise_seed"],
    )
    validation_set = RxRx1ImageListDataset(
        built["val_records"], image_root=image_root, sample_size=sample_size
    )
    test_set = RxRx1ImageListDataset(
        built["test_records"], image_root=image_root, sample_size=sample_size
    )

    class_names = [built["class_names"].get(o, f"class{o}") for o in built["target_orig"]]
    dataset_config = {
        # `apply_open_set_modifications` 会读取它作为 total_classes，然后再改写成
        # len(ind_classes)。所以这里必须是 I + D（目标类总数），不是 I。
        "num_classes": total_classes,
        "all_classes": total_classes,
        "feature_sizes": feature_sizes(sample_size),
        "encoding_dimension": 240,  # 死键（全仓只写不读），仅为兼容保留
        "sample_size": (sample_size, sample_size),
        "class_names": class_names,
        "ind_original_class_ids": ind_orig,
        "discovery_original_class_ids": disc_orig,
        "far_source": (built["far_spec"] or {}).get("source"),
        "far_n_samples": (built["far_spec"] or {}).get("n_samples", 0),
        "split_file": str(split_path),
        "split_name": split["meta"]["name"],
        "initial_per_class": built["initial_per_class"],
        "pool_size": len(built["pool_records"]),
        "n_labeled_initial": len(built["label_idx"]),
    }

    if scenario_converter is None:
        raise ValueError("load_rxrx1 需要注入 scenario_converter（由适配器提供）")

    return (
        training_pool,
        built["label_idx"],
        [built["unlabeled_idx"]],
        validation_set,
        test_set,
        dataset_config,
        scenario_converter,
    )
