# 实验三：RxRx1-OSDAL-DS —— 域偏移下的开放集发现主动学习（最终形态）

> 状态：**roadmap（尚未开始）**。本文件给出最终目标的拆解与接口预留。
> 概述见 [`../README.md`](../README.md) §2.1；两个前置实验见
> [`../exp1_osal/protocol.md`](../exp1_osal/protocol.md) 与
> [`../exp2_domain_shift/design.md`](../exp2_domain_shift/design.md)。

## 目标

把实验一的「OOD 过滤 + 新类发现」与实验二的「跨批次域偏移」**合并**为同一个设定：

> 在未标注池 U 同时存在 **新类**（D）与 **无关数据**（O）、且 **源域与目标域 batch 不一致** 时，
> 主动学习要同时做到：过滤 O、发现 D、并泛化到未见过的实验批次。

记为 **RxRx1-OSDAL-DS**（Open-Set Discovery AL under Domain Shift）。

## 四个综合点（相对前置实验的缺口）

| # | 综合点 | 前置实验的缺口 | 综合做法 |
|---|---|---|---|
| **S1** | role × 域**双层切分** | 实验一只按**类**切 I/D/O；实验二只按**批次**切 train/val/test，池是纯 I | U 按「类 × 域」二维切：I = 已知类×源域，D = 新类（可跨域发现），O = 无关数据 **+ 无关联批次** |
| **S2** | **域鲁棒**的 OOD 阈值 | energy 阈值在**同分布**上拟合才稳；跨域时"源域拟合、目标域失效" | 阈值**按 batch/域分层估计**，或在目标域上在线重估（在线更新 Youden 阈值） |
| **S3** | **域感知**的选择 | SISOMe 的 $d_{in}/d_{out}$ 建立在源批次潜空间上，可能偏向源域；Eq.(4) 只平衡**类**，不平衡**域** | 在 Eq.(4) 之外增加 **batch/域平衡项**（把 class share 换成 domain share）+ 域多样性正则 |
| **S4** | **统一评测口径** | 实验一只报 target-class acc / precision；实验二只报 OOD acc / gen_gap / worst-group | 统一报告：`acc_target` × `gen_gap` × `worst_celltype` × `selection_precision` × `n_discovered` |

## 架构上已经预留的接口（无需推倒重来）

| 预留点 | 位置 | 说明 |
|---|---|---|
| role 配置自由化 | `experiment_config.data_config` | 已是自由 dict，可同时声明 `ind_original_class_ids` / `discovery_original_class_ids` / `far_data_config`，再加一份 `domain_split`（按 `experiment` 划分源/目标域）即可 |
| 映射可扩展 | `AlteredDataset.map`（`ood_detection_datasets.py`） | 现为纯加法映射（InD→`0..C-1`、D→`C..`、O→`-1`）；未来扩成「class-map × domain-map」的直积，其余链路（`OpenDataSetHandler`、`JodaLogic`、指标）不用动 |
| 阈值可替换 | `JodaLogic.estimate_ood_threshold` / `get_ind_candidate_mask` | 独立的两个方法，是注入「域分层/域鲁棒阈值」的最小侵入点 |
| 平衡项可替换 | `JodaLogic.apply_class_balance` | 独立函数，可并列增加一个 `apply_domain_balance` |
| loader 可扩展 | `exp/rxrx1/common/rxrx1_dataset.py`（计划） | 角色/域划分全部由 `data_config` 驱动，新增域划分无需改 loader |

## 建议路线

1. **阶段 1（现在）**：实验一 —— 论文的 OSDAL 在 RxRx1 上复现 + 适配（`exp1_osal/`）。
2. **阶段 2**：实验二 —— 论文之外的"选择策略在域偏移下的可迁移性"压力测试（`exp2_domain_shift/`）。
3. **阶段 3**：综合 —— 复用阶段 1 的 loader/role 机制 + 阶段 2 的域切分，实现 S2/S3 两个改良点。

## 待定问题（开工前需要回答）

- Q1：D 类是否允许跨域（同一新类在源/目标域都出现并联合发现），还是限定在每个域内独立发现？
- Q2：O 的域外部分如何构造——用其他 `cell_type` 的留出实验，还是外部显微数据集？
- Q3：域标签是否在训练时可用（domain-adversarial / 域平衡），还是仅在评测时用于分组？
- Q4：主指标如何同时体现"发现"与"泛化"——是否需要一个新的复合指标（如 `acc_target` 的 worst-group 版本）？
