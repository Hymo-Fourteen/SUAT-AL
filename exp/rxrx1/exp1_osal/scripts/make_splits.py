#!/usr/bin/env python
"""生成并冻结实验一的类别划分（`configs/splits/*.json`）。

用法::

    python exp/rxrx1/exp1_osal/scripts/make_splits.py --root data/wilds/rxrx1_v1.0

默认写出一组划分文件；用 ``--only`` 只生成其中一个。划分一旦用于正式实验就
**不应再改动**（改动等于换了实验设定）。

far-OOD 说明（`protocol.md` §3.3）：

* ``smoke`` 用 F2（即时生成的 ``randn``），不依赖任何外部数据。
* B/A 族默认 ``far_ood: null``（暂不含 far-OOD）。F1（MNIST→PNG 等外部目录）
  在 Stage 1 就绪后用 ``--far-ood dir --far-root <路径> --far-n <数量>`` 重新生成；
  在那之前 B/A 划分可用于「无 far-OOD」的链路验证。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.rxrx1.common.splits import (  # noqa: E402
    build_split_dict,
    far_ood_spec,
    load_metadata,
    save_split,
)

DEFAULT_SPECS = [
    # (文件名, 方案, initial_per_class, far-OOD 来源, far-OOD 数量)
    ("smoke_s0", "smoke", 2, "noise", 200),
    ("B60_s0", "B60", 5, None, 0),
    ("B50_s0", "B50", 5, None, 0),
    ("A60_s0", "A60", 5, None, 0),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/wilds/rxrx1_v1.0", help="RxRx1 数据根目录（含 metadata.csv 与 images/）")
    parser.add_argument("--out", default="exp/rxrx1/exp1_osal/configs/splits", help="输出目录")
    parser.add_argument("--only", default=None, help="只生成指定文件名（不含 .json），如 smoke_s0")
    parser.add_argument("--far-ood", choices=["none", "noise", "dir"], default=None, help="覆盖 B/A 族的 far-OOD 来源")
    parser.add_argument("--far-root", default=None, help="--far-ood dir 时的图片目录")
    parser.add_argument("--far-n", type=int, default=0, help="--far-ood dir 时的样本数")
    parser.add_argument("--far-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    if not root.is_dir():
        print(f"[错误] 数据根目录不存在: {root}", file=sys.stderr)
        return 2

    meta = load_metadata(root)
    print(f"metadata: {len(meta)} 行，{meta['sirna_id'].nunique()} 个类别")

    for name, scheme, ipc, far_source, far_n in DEFAULT_SPECS:
        if args.only and name != args.only:
            continue

        source = far_source
        n = far_n
        if args.far_ood is not None and scheme != "smoke":
            source = None if args.far_ood == "none" else args.far_ood
            n = args.far_n

        spec = None
        if source == "noise":
            spec = far_ood_spec("noise", n_samples=n, seed=args.far_seed)
        elif source == "dir":
            if not args.far_root:
                print(f"[跳过] {name}: --far-ood dir 需要 --far-root", file=sys.stderr)
                continue
            spec = far_ood_spec("dir", n_samples=n, seed=args.far_seed, image_root=args.far_root)

        split = build_split_dict(
            meta,
            scheme,
            seed=args.seed,
            initial_per_class=ipc,
            far_ood=spec,
            metadata_path=str(root / "metadata.csv"),
        )
        path = save_split(split, out_dir / f"{name}.json")

        counts = split["meta"]["counts"]
        print(
            f"[写出] {path}\n"
            f"       方案={scheme}  I={counts['I']}  D={counts['D']}  "
            f"池={counts['pool_images']}(InD {counts['pool_images_inD']} + D {counts['pool_images_D']})  "
            f"L0={counts['L0']}  id_test={counts['id_test_images_experiment']}  "
            f"far-OOD={spec['source'] + ':' + str(spec['n_samples']) if spec else '无'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
