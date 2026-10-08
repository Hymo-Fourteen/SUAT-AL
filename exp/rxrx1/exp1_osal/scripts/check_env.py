#!/usr/bin/env python
"""检查运行实验一所需的环境与数据是否就绪。

用法：
    conda run -n joda-cuda python exp/rxrx1/exp1_osal/scripts/check_env.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]          # .../SUAT-AL
DATA_DIR = REPO_ROOT / "data" / "wilds" / "rxrx1_v1.0"

CORE = ["torch", "torchvision", "numpy", "PIL", "pandas"]
JODA_RUNTIME = [
    "yaml", "sklearn", "tqdm", "mlflow", "segmentation_models_pytorch",
    "torchmetrics", "pyro", "matplotlib", "setuptools", "pkg_resources",
]

ok, warn, bad = [], [], []


def check_module(name: str, *, required: bool = True) -> None:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "ok")
        print(f"  {name:<30} {ver}")
        ok.append(name)
    except Exception as exc:                     # noqa: BLE001
        print(f"  {name:<30} MISSING ({type(exc).__name__})")
        (bad if required else warn).append(name)


def main() -> int:
    print("=" * 72)
    print("Python :", sys.version.split()[0])
    print("Exec   :", sys.executable)
    print("=" * 72)

    print("\n[1] 核心依赖")
    check_module("torch")
    check_module("torchvision")
    for m in ["numpy", "PIL", "pandas"]:
        check_module(m)

    print("\n[2] JODA 运行时依赖")
    for m in JODA_RUNTIME:
        check_module(m, required=(m in {"yaml", "sklearn", "tqdm", "mlflow",
                                        "segmentation_models_pytorch", "torchmetrics"}))

    print("\n[3] CUDA / GPU")
    try:
        import torch
        avail = torch.cuda.is_available()
        print(f"  torch.cuda.is_available()     {avail}")
        print(f"  torch.version.cuda           {torch.version.cuda}")
        print(f"  torch.cuda.get_arch_list()   {torch.cuda.get_arch_list()}")
        if not avail:
            print("  -> 只能用 CPU 跑（冒烟可以，主实验不行）")
            warn.append("cuda")
        else:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"  device                       {name}  (sm_{cap[0]}{cap[1]})")
            # 真跑一次，确认该架构有可用 kernel
            x = torch.randn(2048, 2048, device="cuda")
            y = (x @ x).sum().item()
            torch.cuda.synchronize()
            print(f"  GPU matmul test              OK (sum={y:.3f})")
            print(f"  VRAM total                   {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
            ok.append("gpu-kernel")
    except Exception as exc:                     # noqa: BLE001
        print(f"  ❌ {type(exc).__name__}: {exc}")
        bad.append("cuda")

    print("\n[4] RxRx1 数据")
    if DATA_DIR.exists():
        print(f"  {DATA_DIR}  (-> {os.path.realpath(DATA_DIR)})")
        meta = DATA_DIR / "metadata.csv"
        img = DATA_DIR / "images"
        print(f"  metadata.csv : {'OK' if meta.is_file() else 'MISSING'}"
              f"  ({sum(1 for _ in meta.open()) - 1 if meta.is_file() else 0} 行)")
        n_exp = len(list(img.iterdir())) if img.is_dir() else 0
        print(f"  images/      : {n_exp} 个实验目录")
        (ok if (meta.is_file() and n_exp) else bad).append("rxrx1-data")
    else:
        print(f"  ❌ 找不到 {DATA_DIR}")
        bad.append("rxrx1-data")

    print("\n[5] JODA 代码")
    joda_src = REPO_ROOT / "JODA" / "src" / "joda_al"
    print(f"  {joda_src} : {'OK' if joda_src.is_dir() else 'MISSING'}")
    (ok if joda_src.is_dir() else bad).append("joda-src")

    print("\n" + "=" * 72)
    print(f"结论: {len(ok)} 项通过, {len(warn)} 项警告, {len(bad)} 项失败")
    if warn:
        print("警告:", ", ".join(warn))
    if bad:
        print("失败:", ", ".join(bad))
        return 1
    print("环境可用于实验一 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
