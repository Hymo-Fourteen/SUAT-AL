"""RxRx1 的类别角色划分与固定切分。

本模块**纯离线**：只依赖 `pandas`，不 import `joda_al`，因此可以独立运行来生成
并冻结划分文件（`configs/splits/*.json`），再由 loader 读取。

设计要点（与 `protocol.md` §3.1–3.3 对应）：

* WILDS 官方 split 由 `metadata.csv` 的 `dataset` 列给出（train / val / test），
  其内部再用 `site` 列区分同分布与域偏移：``train`` 的 site 1 是 AL 池，
  site 2 是同分布测试集。这一点 `dataset` 列本身并不体现，必须显式派生。
* 类别角色分两族（`protocol.md` §3.2）：
  - ``B<ratio>``（主设定）：比例作用在 1,108 个 treatment 类上，31 个控制类
    （30 正控 + ``EMPTY`` 负控）固定放入 InD。
  - ``A<ratio>``（论文式对照）：直接对 1,139 个 ``sirna_id`` 连续取前 ratio。
* 池只包含本实验涉及的类别（``InD ∪ D``）。完整方案下二者之和就是全部 1,139 类，
  但小规模冒烟方案会显著缩小类别集，此时必须过滤掉无关类别的图像，
  否则池里会残留「无角色」的样本。
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

# --------------------------------------------------------------------------- #
# 数据集常量（已用 metadata.csv 核实，见 notes.md）
# --------------------------------------------------------------------------- #
N_CLASSES = 1139
N_TREATMENT = 1108  # sirna_id 0..1107，每类在池中恰好 33 张
TREATMENT_IDS: List[int] = list(range(0, N_TREATMENT))
CONTROL_IDS: List[int] = list(range(N_TREATMENT, N_CLASSES))  # 1108..1138
NEGATIVE_CONTROL_ID = 1138  # sirna == "EMPTY"
POSITIVE_CONTROL_IDS: List[int] = list(range(1108, 1138))

# 派生 split 名 -> (dataset 列取值, site 列取值或 None)
SPLIT_RULES: Dict[str, tuple] = {
    "pool": ("train", 1),  # AL 池，40,612 张
    "id_test": ("train", 2),  # 同分布测试集，40,612 张
    "ood_val": ("val", None),  # 域偏移验证，9,854 张（4 个实验批次为 OOD）
    "ood_test": ("test", None),  # 域偏移测试，34,432 张（14 个实验批次为 OOD）
}

IMAGE_REL_TEMPLATE = "images/{experiment}/Plate{plate}/{well}_s{site}.png"

# 冒烟方案：仅取极少类别，让 CPU 上能在分钟级跑通全链路。
SMOKE_N_IND = 6
SMOKE_N_DISC = 4


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def load_metadata(root: str | Path, metadata_csv: str = "metadata.csv") -> pd.DataFrame:
    """读取并校验 RxRx1 的 metadata。"""
    root = Path(root)
    csv_path = root / metadata_csv
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"找不到 {csv_path}。请确认 RxRx1 数据已下载，并已建立 "
            f"`data/wilds/rxrx1_v1.0 -> <真实路径>` 软链接。"
        )
    meta = pd.read_csv(csv_path)
    required = {"site_id", "cell_type", "dataset", "experiment", "plate", "well", "site", "well_type", "sirna", "sirna_id"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata.csv 缺少列: {sorted(missing)}")
    return meta


def class_table(meta: pd.DataFrame) -> pd.DataFrame:
    """返回以 ``sirna_id`` 为索引的类别表（sirna / well_type）。"""
    table = meta.groupby("sirna_id", sort=True).agg(
        sirna=("sirna", "first"), well_type=("well_type", "first")
    )
    if len(table) != N_CLASSES:
        raise ValueError(f"期望 {N_CLASSES} 个类别，实际 {len(table)}")
    return table


def split_indices(meta: pd.DataFrame) -> Dict[str, Sequence[int]]:
    """按 WILDS 规则派生 4 个 split 的行索引。"""
    out: Dict[str, Sequence[int]] = {}
    for name, (dataset_value, site_value) in SPLIT_RULES.items():
        mask = meta["dataset"] == dataset_value
        if site_value is not None:
            mask &= meta["site"] == site_value
        out[name] = meta.index[mask]
    return out


# --------------------------------------------------------------------------- #
# 类别角色
# --------------------------------------------------------------------------- #
_SCHEME_RE = re.compile(r"^([AaBb])(\d+)$")


def make_class_roles(
    meta: pd.DataFrame,
    scheme: str,
    *,
    seed: int = 0,
    pick: str = "first",
) -> Dict[str, object]:
    """按方案名生成 InD / Discovery 类别划分。

    参数
    ----
    scheme : ``"B60"`` / ``"B50"`` / ``"A60"`` / ``"smoke"``
        字母为族（见模块 docstring），数字为百分比。
    seed : 随机挑选 treatment 类（``pick="random"``）时使用的种子。
    pick : ``"first"``（按 ``sirna_id`` 连续，论文惯例）或 ``"random"``（稳健性检查）。
    """
    if scheme == "smoke":
        # 冒烟方案：类别数极少（远小于 B/A 族），池也相应被裁到这几类，
        # 目的是在 CPU 上分钟级跑通全链路，而非复现论文口径。
        ind = TREATMENT_IDS[:SMOKE_N_IND]
        disc = TREATMENT_IDS[SMOKE_N_IND : SMOKE_N_IND + SMOKE_N_DISC]
        family, ratio = "smoke", None
    else:
        match = _SCHEME_RE.match(scheme)
        if match is None:
            raise ValueError(
                f"无法解析方案名 {scheme!r}；期望形如 'B60'、'A50' 或 'smoke'。"
            )
        family = match.group(1).upper()
        ratio = int(match.group(2)) / 100.0
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"比例必须在 (0, 1) 内，得到 {ratio}")
        if family == "A":
            k = round(N_CLASSES * ratio)
            ind = list(range(k))
        else:
            k = round(N_TREATMENT * ratio)
            if pick == "random":
                picked = sorted(random.Random(seed).sample(TREATMENT_IDS, k))
            elif pick == "first":
                picked = TREATMENT_IDS[:k]
            else:
                raise ValueError(f"pick 只能是 'first' 或 'random'，得到 {pick!r}")
            ind = sorted(picked + CONTROL_IDS)
        disc = [c for c in range(N_CLASSES) if c not in set(ind)]

    if not disc:
        raise ValueError(f"方案 {scheme} 没有留下任何 Discovery 类别")

    table = class_table(meta)
    names = {int(cid): str(table.loc[cid, "sirna"]) for cid in range(N_CLASSES)}
    return {
        "scheme": scheme,
        "family": family,
        "inD_ratio": ratio,
        "pick": pick,
        "ind_original_class_ids": [int(c) for c in ind],
        "discovery_original_class_ids": [int(c) for c in disc],
        "control_original_class_ids": [int(c) for c in CONTROL_IDS],
        "class_names": names,
    }


# --------------------------------------------------------------------------- #
# 样本清单
# --------------------------------------------------------------------------- #
def image_rel_path(row) -> str:
    """由 metadata 的一行构造图片相对路径。

    模板已用 400 个随机样本验证过（notes.md 记录）。
    """
    return IMAGE_REL_TEMPLATE.format(
        experiment=row["experiment"], plate=row["plate"], well=row["well"], site=row["site"]
    )


def pool_frame(meta: pd.DataFrame, class_ids: Iterable[int]) -> pd.DataFrame:
    """池 split 中属于给定类别的行（保持 metadata 原顺序）。"""
    index = split_indices(meta)["pool"]
    frame = meta.loc[index]
    return frame[frame["sirna_id"].isin(set(class_ids))]


def select_initial_labeled(
    pool: pd.DataFrame,
    ind_class_ids: Sequence[int],
    initial_per_class: int,
    *,
    seed: int = 0,
) -> Dict[int, List[str]]:
    """为每个 InD 类确定初始标注样本（返回类 id -> 相对路径列表）。

    选择必须是**确定性的**且与全局随机状态无关，否则无法复现
    （见 notes.md 中关于初始标注集可复现性的记录）。这里用
    ``random.Random(seed * 100003 + class_id)`` 做每类独立抽样。
    """
    columns = ["experiment", "plate", "well", "site"]
    out: Dict[int, List[str]] = {}
    for cid in ind_class_ids:
        rows = pool[pool["sirna_id"] == cid]
        n = len(rows)
        if n < initial_per_class:
            raise ValueError(
                f"类别 {cid} 在池中只有 {n} 张，少于 initial_per_class={initial_per_class}"
            )
        rel_paths = [image_rel_path(r) for _, r in rows[columns].iterrows()]
        if n > initial_per_class:
            rng = random.Random(seed * 100003 + int(cid))
            rel_paths = sorted(rng.sample(rel_paths, initial_per_class))
        out[int(cid)] = sorted(rel_paths)
    return out


def far_ood_spec(
    source: str,
    *,
    n_samples: int,
    seed: int = 0,
    image_root: str | None = None,
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> Dict[str, object]:
    """构造 far-OOD 描述（`protocol.md` §3.3 的 F1/F2）。

    * ``source="noise"`` —— F2，按索引种子即时生成 ``randn``，不占磁盘。
    * ``source="dir"``   —— F1，从目录递归收集图片后按种子抽样。
    """
    if source not in {"noise", "dir"}:
        raise ValueError(f"far-OOD source 只能是 'noise' 或 'dir'，得到 {source!r}")
    spec: Dict[str, object] = {"source": source, "n_samples": int(n_samples), "seed": int(seed)}
    if source == "dir":
        if not image_root:
            raise ValueError("source='dir' 时必须提供 image_root")
        spec["image_root"] = str(image_root)
        spec["extensions"] = list(extensions)
    return spec


def list_far_ood_images(spec: Dict[str, object]) -> List[str]:
    """按 spec 列出 far-OOD 图片的**绝对路径**（确定性：先排序再按种子抽样）。"""
    root = Path(str(spec["image_root"]))
    if not root.is_dir():
        raise FileNotFoundError(f"far-OOD 图片目录不存在: {root}")
    extensions = tuple(str(e).lower() for e in spec.get("extensions", [".png"]))
    paths = sorted(
        str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )
    n = int(spec["n_samples"])
    if len(paths) < n:
        raise ValueError(f"far-OOD 目录只有 {len(paths)} 张图，不足所需的 {n} 张")
    if len(paths) > n:
        paths = sorted(random.Random(int(spec["seed"])).sample(paths, n))
    return paths


# --------------------------------------------------------------------------- #
# 冻结
# --------------------------------------------------------------------------- #
def build_split_dict(
    meta: pd.DataFrame,
    scheme: str,
    *,
    seed: int = 0,
    pick: str = "first",
    initial_per_class: int = 5,
    far_ood: Dict[str, object] | None = None,
    metadata_path: str | None = None,
) -> Dict[str, object]:
    """生成一份完整的、可冻结的切分描述。"""
    roles = make_class_roles(meta, scheme, seed=seed, pick=pick)
    ind = roles["ind_original_class_ids"]
    disc = roles["discovery_original_class_ids"]
    pool = pool_frame(meta, list(ind) + list(disc))
    splits = split_indices(meta)

    initial_labeled = select_initial_labeled(
        pool, ind, initial_per_class, seed=seed
    )
    target_ids = set(ind) | set(disc)
    scope = {name: meta.loc[idx] for name, idx in splits.items()}
    scoped = {name: frame[frame["sirna_id"].isin(target_ids)] for name, frame in scope.items()}
    counts = {
        "I": len(ind),
        "D": len(disc),
        "n_classes_total": N_CLASSES,
        "pool_images": int(len(pool)),
        "pool_images_inD": int((pool["sirna_id"].isin(set(ind))).sum()),
        "pool_images_D": int((pool["sirna_id"].isin(set(disc))).sum()),
        "L0": int(initial_per_class * len(ind)),
        # 下列 *_experiment 是**只含本实验类别**的口径（loader 实际使用的集合）；
        # 不带后缀的是 WILDS 全量 split 规模，仅供对照。
        "id_test_images_experiment": int(len(scoped["id_test"])),
        "ood_val_images_experiment": int(len(scoped["ood_val"])),
        "id_test_images_all_classes": int(len(splits["id_test"])),
        "ood_val_images_all_classes": int(len(splits["ood_val"])),
        "ood_test_images_all_classes": int(len(splits["ood_test"])),
    }
    name = f"{scheme}_s{seed}" if pick == "first" else f"{scheme}_{pick}_s{seed}"
    return {
        "meta": {
            "name": name,
            "scheme": scheme,
            "family": roles["family"],
            "inD_ratio": roles["inD_ratio"],
            "pick": pick,
            "seed": seed,
            "initial_per_class": initial_per_class,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metadata_csv": metadata_path,
            "n_metadata_rows": int(len(meta)),
            "split_rules": {k: list(v) for k, v in SPLIT_RULES.items()},
            "counts": counts,
        },
        "class_roles": {
            "ind_original_class_ids": ind,
            "discovery_original_class_ids": disc,
            "control_original_class_ids": roles["control_original_class_ids"],
        },
        "class_names": roles["class_names"],
        "far_ood": far_ood,
        "initial_labeled": {str(k): v for k, v in initial_labeled.items()},
    }


def save_split(split: Dict[str, object], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, ensure_ascii=False, indent=1, sort_keys=False)
        handle.write("\n")
    return path


def load_split(path: str | Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
