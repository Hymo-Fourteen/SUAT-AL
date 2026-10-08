# exp/rxrx1/common —— 多实验共用代码

本目录存放**与 JODA 解耦**、在多个 RxRx1 实验间复用的实现。

## 规划内容

| 文件（计划） | 作用 | 对应设计 |
|---|---|---|
| `rxrx1_dataset.py` | `RxRx1ImageListDataset` + 元数据/路径解析（`metadata.csv` → `(path, label)`） | exp1 §2.9 |
| `splits.py` | 类角色划分（InD / discovery）与域划分（源/目标 experiment）生成器；输出可复现的 class-id json | exp1 §2.9、exp2 §3.3、exp3 |
| `transforms.py` | `InstanceStandardize`、`Rotate90` 等 RxRx1 专用变换 | exp1 §2.11 |
| `eval_offline.py` | 在 checkpoint 上离线补算 `acc_id` / `acc_ood` / `worst_celltype` / `f1_macro` | exp1 §2.4、exp2 §3.3 |
| `memmap.py` | 预转 `uint8` memmap 缓存（24.7 GB）与读索引 | exp1 §2.14 |
| `joda_adapter.py` | 薄适配：把上面的实现注册进 JODA（loader factory / trans_map / `ALL_DATASETS`） | exp1 §2.10 |

## 约定

- **不 import `joda_al`**：保持纯 Python / PyTorch，便于单测与跨实验复用。
- JODA 侧只保留一个 thin shim：
  `JODA/src/joda_al/data_loaders/classification/rxrx1_data_loader.py`
  → 导入本目录的实现并完成 4 处注册。
- 所有路径与划分参数由 `experiment_config.data_config` 驱动，不写死在本目录里。
