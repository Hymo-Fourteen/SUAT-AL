# 实验二：RxRx1 原生任务上的跨批次主动学习（域偏移纯 AL）

> 本文件是实验二的**完整设计与实施规格**。
> 共用背景（论文要点 §0、RxRx1 数据事实 §1、代码改动总表、待决策项）见 [`../README.md`](../README.md)。
> 实验一见 [`../exp1_osal/protocol.md`](../exp1_osal/protocol.md)（协议）与 [`../exp1_osal/notes.md`](../exp1_osal/notes.md)（记录）。

---


## 3.1 要回答的问题

1. 把 JODA 的选择策略（SISOMe + 类平衡）用于**纯 InD、但存在强 batch effect** 的场景，
   能否比 Random/Entropy/BADGE/CoreSet 更省标注？
2. 域偏移下，AL 是缓解还是**加剧**了 ID/OOD 的泛化差距（gen gap）？
3. JODA 的类平衡项（Eq.4）在 1139 类、U2OS 仅占 9% 的强不平衡下是否更关键？

## 3.2 ⚠️ 必须先明确的语义

**实验二里 JODA 的「过滤」组件是失效的、会被自动跳过**：

- 池是纯 InD（无 far-OOD）；
- `estimate_ood_threshold()` 在 labeled 池里只有单一类别时直接 `return`（`optimal_threshold=None`）；
- `get_ind_candidate_mask()` 在 `optimal_threshold is None` 时返回**全 True** → 不过滤；
- `data_scenario: standard` 更不会构造 near/far 类。

⇒ **实验二实际评估的是 JODA 的「选择 + 类平衡」子模块**（≈ SISOMe-BC），
不是完整的 JODA。论文并没有覆盖「域偏移下的纯 AL」这一设定，
所以实验二是**论文之外的补充评测**，结论要按「选择策略在域偏移下的可迁移性」来陈述，不能说成「JODA 全流程」。

## 3.3 协议设计

| 项目 | 设计 |
|---|---|
| AL 池 | `train`(site1) 40,612 |
| 验证集 | `val`(4 个 OOD 实验) 9,854 → 早停/选模型 |
| 主测试 | `test`(14 个 OOD 实验) 34,432 → `acc_ood` |
| 辅测试 | `id_test`(site2) 40,612 → `acc_id`（离线在 checkpoint 上补算） |
| 初始 labeled | 按类分层 **1 张/类 = 1,139**（随机 1139 只能覆盖 ~63% 的类，见 §1） |
| query | **1,139 / 轮**（≈1 张/类/轮） |
| cycles | **9**（最终 10,251 张 ≈ 池的 25%） |
| 训练 | `strategy: retrain`；`Resnet18T` @256，ImageNet 预训练；Adam 1e-3，wdcay 1e-5；实例标准化 + 随机 90° 旋转 + 水平翻转 |
| 指标 | `acc_ood`（主）、`acc_id`、`gen_gap = acc_id - acc_ood`、`acc_worst_celltype`、`acc_avg_celltype`、`f1_macro`、`query_time`、`n_labeled` |
| seeds | 0/1/2 |

## 3.4 方法对照

| 方法 | 配置 | 说明 |
|---|---|---|
| **Random** | `method_type: Random` | 已在 `static_querys.py` 注册，开箱可用 |
| **Entropy** | `method_type: Ent` | 经典不确定性 |
| **LLoss** | `method_type: LLoss` | 论文中 CIFAR 上的强基线 |
| **Badge** | `method_type: Badge` | 多样性+不确定性；1139 类下算梯度可能吃显存 |
| **CoreSet** | `method_type: CoreSet`（若注册名不同需确认） | 纯多样性 |
| **LfOSA** | `method_type: LfOSA` | 开放集 AL 方法（池纯 InD 时应退化） |
| **Joda** | `method_type: Joda`，`data_scenario: standard`，`apply_ood_filter: false` | 选择+类平衡 |

## 3.5 消融

| 编号 | 消融 | 目的 |
|---|---|---|
| E2-1 | Joda：`acl_balancing` on/off | 类平衡项在 1139 类不平衡下的作用 |
| E2-2 | 初始集：分层 1/类 vs 随机 1139 | 验证分层初始化必要性 |
| E2-3 | `pretrained` on/off | 小预算下预训练作用 |
| E2-4 | 实例标准化 vs 全局常数归一化 | 验证 §1 的归一化结论 |
| E2-5 | 旋转增强 on/off | 增强对 batch shift 的作用 |
| E2-6 | 8 轮 vs 15 轮 | 曲线是否饱和 |

## 3.6 配置草稿

```yaml
experiment_config:
  method_type: Joda          # 对照时改 Random / Ent / LLoss / Badge
  query_scenario: Pool
  data_scenario: standard    # ★ 纯 AL，不构造 near/far
  dataset: rxrx1
  dataset_path: ./data/wilds
  static_configuration: ta
  cycles: 9
  query_size: 1139
  is_percent: false
  over_selection: 1.0
  task: classification
  engine: pytorch
  seed: 0
training_config:
  loss_func: CrossEntropy
  model: Resnet18T
  pretrained: true
  num_epochs: 30
  batch_size: 64
  optimizer: Adam
  lr: 0.001
  wdcay: 0.00001
  num_workers: 8
  early_stopping: val_acc
  eval_frequencey: 5
  acl_balancing: true
  apply_ood_filter: false    # 纯 InD 池，过滤无意义
```

---

