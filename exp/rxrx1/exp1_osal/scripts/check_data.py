#!/usr/bin/env python
"""离线自检 RxRx1 数据链路（**不启动 JODA**）。

在跑完整实验之前先用它把「划分 → 池构造 → 标签空间 → 变换」这条链路验证掉，
避免把数据问题误判成 JODA 的问题。

用法::

    python exp/rxrx1/exp1_osal/scripts/check_data.py --split smoke_s0
    python exp/rxrx1/exp1_osal/scripts/check_data.py --split B60_s0 --sample-images 3

退出码非 0 表示有断言失败。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from exp.rxrx1.common.rxrx1_dataset import (  # noqa: E402
    NOISE_KIND,
    RxRx1ImageListDataset,
    build_pool_and_splits,
)
from exp.rxrx1.common.splits import load_split  # noqa: E402
from exp.rxrx1.common.transforms import EVAL_TRANSFORM, TRAIN_TRANSFORM, Rotate90  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  [ok]   {message}")
    else:
        print(f"  [FAIL] {message}")
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/wilds/rxrx1_v1.0")
    parser.add_argument("--splits-dir", default="exp/rxrx1/exp1_osal/configs/splits")
    parser.add_argument("--split", default="smoke_s0", help="划分文件名（不含 .json）")
    parser.add_argument("--sample-images", type=int, default=2, help="实际解码几张图检查变换")
    parser.add_argument("--verify-files", action="store_true", help="校验所有图片是否存在（全量方案较慢）")
    args = parser.parse_args()

    root = Path(args.root)
    split_path = Path(args.splits_dir) / f"{args.split}.json"
    print(f"划分文件: {split_path}")
    split = load_split(split_path)
    meta = split["meta"]
    print(f"方案 {meta['scheme']}  I={meta['counts']['I']}  D={meta['counts']['D']}  "
          f"initial_per_class={meta['initial_per_class']}  far={split.get('far_ood')}")

    print("\n[1] 构造池与切分")
    built = build_pool_and_splits(
        split=split, image_root=root, verify_files=args.verify_files,
    )
    ind, disc = built["ind_orig"], built["disc_orig"]
    n_ind, n_disc = len(ind), len(disc)
    pool_records = built["pool_records"]
    label_idx = built["label_idx"]

    check(len(label_idx) == n_ind * meta["initial_per_class"],
          f"初始标注集大小 = {len(label_idx)}（期望 I x ipc = {n_ind * meta['initial_per_class']}）")
    check(label_idx == list(range(len(label_idx))), "初始标注集是池的**前缀**（label_idx == range）")
    check(len(built["unlabeled_idx"]) == len(pool_records) - len(label_idx),
          f"未标注集大小 = {len(built['unlabeled_idx'])}")

    print("\n[2] 池的标签空间")
    pool_ds = RxRx1ImageListDataset(pool_records, image_root=root, noise_seed=built["noise_seed"])
    targets = pool_ds.targets
    inD = [t for t in targets if 0 <= t < n_ind]
    D = [t for t in targets if n_ind <= t < n_ind + n_disc]
    far = [t for t in targets if t < 0]
    check(len(inD) + len(D) + len(far) == len(targets), "所有池样本都落在这三类标签里")
    check(len(far) == (built["far_spec"] or {}).get("n_samples", 0),
          f"far-OOD 样本数 = {len(far)}")
    check(all(t == -1 for t in far), "far-OOD 标签全为 -1")
    check(len({t for t in inD}) == n_ind, f"InD 标签覆盖 0..{n_ind - 1}（实际 {len(set(inD))} 个）")
    check(len({t for t in D}) == n_disc, f"D 标签覆盖 {n_ind}..{n_ind + n_disc - 1}（实际 {len(set(D))} 个）")

    initial_labels = {targets[i] for i in label_idx}
    check(initial_labels == set(range(n_ind)), "初始标注集只含 InD 标签")

    print("\n[3] 验证/测试集")
    for name, records in (("ood_val", built["val_records"]), ("id_test", built["test_records"])):
        labels = {r[1] for r in records}
        check(labels <= set(range(n_ind + n_disc)),
              f"{name}: {len(records)} 张，标签范围 ⊆ [0, {n_ind + n_disc})")
        check(len(labels) == n_ind + n_disc,
              f"{name}: 覆盖全部 {n_ind + n_disc} 个目标类")

    print("\n[4] 变换")
    if args.sample_images > 0:
        file_indices = [i for i, r in enumerate(pool_records) if r[2] != NOISE_KIND][: args.sample_images]
        for i in file_indices:
            image, target = pool_ds[i]
            tensor = EVAL_TRANSFORM(image)
            check(tuple(tensor.shape) == (3, 256, 256), f"eval 变换输出 shape = {tuple(tensor.shape)}")
            check(tensor.dtype == torch.float32, f"eval 变换 dtype = {tensor.dtype}")
            check(abs(float(tensor.mean())) < 1e-4, f"逐图标准化后均值 ≈ 0（{float(tensor.mean()):.2e}）")
            check(abs(float(tensor.std()) - 1.0) < 1e-2, f"逐图标准化后标准差 ≈ 1（{float(tensor.std()):.4f}）")
            augmented = TRAIN_TRANSFORM(tensor)
            check(tuple(augmented.shape) == (3, 256, 256) and augmented.dtype == torch.float32,
                  "训练变换对已标准化张量可用（tensor-safe）")
            check(tuple(Rotate90()(tensor).shape) == (3, 256, 256), "Rotate90 对张量可用")
            break

    noise_idx = next((i for i, r in enumerate(pool_records) if r[2] == NOISE_KIND), None)
    if noise_idx is not None:
        a, b = pool_ds[noise_idx][0], pool_ds[noise_idx][0]
        check(torch.equal(a, b), "噪声 far-OOD 逐索引确定性（多次取同一样本完全一致）")
        check(tuple(EVAL_TRANSFORM(a).shape) == (3, 256, 256), "噪声样本可走同一套 eval 变换")

    print("\n" + "=" * 60)
    if failures:
        print(f"失败 {len(failures)} 项")
        return 1
    print("数据链路自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
