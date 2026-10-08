# RxRx1 × JODA 实验总览

本目录是 **RxRx1 相关全部实验**的统一入口（设计、配置、脚本、产物）。

- 代码库：`JODA/`（JODA 主动学习框架）
- 数据：`data/wilds/rxrx1_v1.0`（软链接 → `/data/yuxuan/rxrx1_v1.0`，7.2 GB）
- 论文：`Schmidt 等 - 2025 - Joint Out-of-Distribution Filtering and Data Discovery Active Learning.pdf`

## 目录结构

```
exp/rxrx1/
├── README.md                 # 本文件：总览 + 共用事实/约定 + 代码放置规则
├── common/                   # 多个 RxRx1 实验共用代码（与 JODA 解耦）
├── exp1_osal/                # 实验一：OSDAL（类发现 + OOD 过滤）★ 已规范化
│   ├── protocol.md           #   ← 规范实验协议（给人看）
│   ├── notes.md              #   ← 全量记录/坑点/实现细节（给 AI 看）
│   └── configs/ scripts/ results/
├── exp2_domain_shift/        # 实验二：跨批次域偏移纯 AL
│   ├── design.md
│   └── configs/ scripts/ results/
└── exp3_combined/            # 最终综合形态：域偏移下做 OSDAL
    ├── design.md             #   ← roadmap（见本文件 §2.1）
    └── configs/ scripts/ results/
```

## 实验索引

| 目录 | 实验 | 状态 | 设计文档 |
|---|---|---|---|
| `exp1_osal/` | 实验一：RxRx1-OSDAL（类发现 + OOD 过滤） | 协议 v0.1，待实现 | [protocol.md](exp1_osal/protocol.md) · [notes.md](exp1_osal/notes.md) |
| `exp2_domain_shift/` | 实验二：RxRx1 原生任务跨批次纯 AL | 设计完成，待实现 | [design.md](exp2_domain_shift/design.md) |
| `exp3_combined/` | 最终形态：域偏移下的 OSDAL（RxRx1-OSDAL-DS） | roadmap | [design.md](exp3_combined/design.md) |

## 代码放置约定

| 类型 | 位置 | 原因 |
|---|---|---|
| 多实验共用（数据集读取、角色/域切分、变换、离线评测） | `exp/rxrx1/common/` | 与 JODA 解耦，可独立复用/测试 |
| 单实验专属（配置、启动/扫描脚本、结果分析、绘图） | `exp/rxrx1/<exp>/scripts/`、`configs/` | 实验隔离，互不干扰 |
| 运行产物（CSV / TensorBoard / checkpoint / 图） | `exp/rxrx1/<exp>/results/` | 不污染代码目录 |
| **必须被 JODA import 的最小接入**（loader 适配 + 4 处注册） | `JODA/src/joda_al/...` | JODA 用 `pkgutil` 自动发现模块，必须在包路径下 |

> JODA 侧只保留「薄适配层」（把 `exp/rxrx1/common/` 的实现导入并注册），
> 真正的实验逻辑全部在 `exp/rxrx1/` 下维护。

## 编号说明

原 `doc/rxrx1_joda_experiment_design.md` 已按实验拆分：

- 原 §2（实验一）→ 已进一步拆为 [`exp1_osal/protocol.md`](exp1_osal/protocol.md)（规范协议）
  与 [`exp1_osal/notes.md`](exp1_osal/notes.md)（全量记录）
- 原 §3（实验二）→ [`exp2_domain_shift/design.md`](exp2_domain_shift/design.md)
- 原 §0 / §1 / §4 / §5 / §6 / 附录 → 保留在本文件（下方）

因此下文若出现「§2.x / §3」的引用，分别指上述两个文件。

---
## 0. 论文要点回顾（只摘与设计相关的）

**OSDAL（Open-Set Discovery Active Learning）场景**：未标注池 U 由三类数据混合而成：

| 记号 | 含义 | 论文中的来源 |
|---|---|---|
| **I**（3a） | 已知 InD 类 | 目标数据集的一部分类 |
| **D**（3b） | 待发现的新类（near-OOD） | 目标数据集剩下的类 |
| **O**（3c） | 无关的远 OOD（far-OOD） | 另一个数据集 |

**JODA 三阶段**（论文 Fig. 2）：

1. **训练（I）**：用 labeled 池里的 InD 部分算 CE，用被误选进来的 OOD 部分算 Outlier-Exposure（OE）损失，
   $\mathcal{L}(b)=\mathcal{L}_{CE}(b_{InD})+\lambda_{OE}\cdot\mathcal{L}_{OE}(b_{OOD})$，$\lambda_{OE}=0.5$。
   目标是把 I/D 在特征空间里聚得更近、把 O 推远。
2. **过滤（II）**：在 labeled 池上对 energy score $E(x)=-\log\sum_i e^{f(x)_i}$ 做 ROC，
   取 **Youden's J 最大**的阈值 $t_{opt}$；$E<t_{opt}$ 视为「InD 或可发现」，否则为 OOD。
   **第 0 轮 labeled 池只有 InD，故跳过过滤**（论文与代码一致）。
3. **选择（III）**：用 SISOMe 分数 $\hat m(x)=\min(m_{avg},1)E(x)+\max(1-m_{avg},0)Q(x)$，
   其中 $Q=d_{in}/d_{out}$，$m_{avg}=\frac{1}{|L|}\sum \frac{d_{in}}{d_{out}}$；
   再叠加 Eq.(4) 的类平衡项 $\hat m_b=\hat m+b_f(\arg\max f(x))$ 后取 top-$q$。

**类别提升阈值 $t_e$**：D 中某类在 labeled 池内累计样本数 $\ge t_e$ 时提升为 InD，
论文取 CIFAR-10=100、CIFAR-100=50、TinyImageNet=25。

**论文实验设定**：3 个 InD 数据集（CIFAR-10/100、TinyImageNet）；**按类别 id 顺序取前 60% 作 InD、
后 40% 作可发现类**；OOD 取 random noise / MNIST / Places365 / ImageNetC-800；
初始 labeled 池 CIFAR-10=1000、CIFAR-100=2000；query 1000(C10)/2500(C100,TIN)；
9 个 cycle；ResNet18 每轮**从零重训 200 epoch**；3 seeds（部分 2 seeds）。
**指标**：(a) 目标类 C+K 上的准确率（未发现类按 0 计），(b) selection precision（I、D 为正，O 为负），(c) 识别出的类别数。

**代码实现对应**（已核对）：

| 论文 | 代码 |
|---|---|
| 场景 | `data_scenario: osal-extending`（`defintions.py`） |
| InD 类 | `experiment_config.ind_classes`；D 自动 = 其余全部类 |
| t_e | `experiment_config.expanding_th`，在 `OpenDataSetHandler.update_data_pool()` 中提升类 |
| far-OOD | `far_dataset` + `far_classes=[-1]`，在 `apply_open_set_modifications()` 里拼到池尾 |
| 标签映射 | `AlteredDataset`：InD→`0..C-1`，D→`C..C+K-1`，O→`-1` |
| OE 损失 | `loss_func: OutlierExposure` + `lambda_oe` |
| 过滤 | `Joda_logic.estimate_ood_threshold()` / `get_ind_candidate_mask()` |
| 选择 | `Joda_logic.calculate_greedy()`（SISOMe）+ `apply_class_balance()`（Eq.4） |
| (a) 指标 | `ClassificationHandler`：`avg_acc = (num_classes/all_classes) × balanced_acc(InD)` |
| (b) 指标 | `sample_selection.filter_osr()` → `Selection/Precision`、`Recall_Lit` |
| (c) 指标 | `OpenDataSetHandler`：`ind_classes` 增长 + 日志 `New classes: ...` |

> ⚠️ 注意：代码里 (a) 是 **(C_t/K_total) × InD balanced accuracy** 的近似（假设未发现类准确率为 0），
> 与论文口径一致但更粗；建议离线再补一个严格版（逐类统计）。

---

## 1. RxRx1 关键事实（实测，load-bearing）

### 1.1 规模与划分

| 项目 | 值 |
|---|---|
| site 级样本总数 | **125,510**（`metadata.csv` 125,510 行；7.2 GB） |
| 类别数 | **1,139**（`sirna_id` 0–1138 连续） |
| 类别构成 | 1,108 treatment + 30 positive control(1108–1137) + 1 negative control/EMPTY(1138) |
| experiment 数 | 51；每实验覆盖 **1121–1139** 类（≈全类） |
| cell_type | HUVEC / RPE / HEPG2 / U2OS（每类均覆盖全部 1139 类） |

WILDS 划分（`dataset` 列 + `site` 列；**`metadata.csv` 的 `dataset` 只有 train/val/test，
`id_test` 需自行拆：`dataset==train & site==2`**）：

| split | 样本 | experiment 数 | 备注 |
|---|---|---|---|
| `train` (site 1) | **40,612** | 33 | AL 池 |
| `id_test` (`train` & site 2) | **40,612** | 33（同 train） | 同分布测试 |
| `val` | 9,854 | 4 | **OOD 实验**，早停/选模型 |
| `test` | 34,432 | 14 | **OOD 实验**，主指标 |

实测 **train ∩ val = train ∩ test = ∅**（实验零重叠）→ 这是 RxRx1 的域偏移来源。

### 1.2 池内每类样本数（决定预算与 $t_e$ 的硬约束）

`train(site1)`：**min 29 / mean 35.7 / median 33 / max 153**。

- **1,108 个 treatment 类几乎是均匀的 31–33 张/类**；
- 31 个 control 类是 131–153 张/类（约 4× 过采样）。

> 🔴 **与 CIFAR-100 最大的不同**：CIFAR-100 每类 500 张，论文 $t_e=50$、每轮 query 2500。
> RxRx1 每类只有 **~33 张**，若照搬论文的 $t_e=50$ 与 query=2500，
> 会把池抽干且**没有任何 D 类能被提升**。因此 $t_e$、初始池、query 必须按池规模等比重设。

### 1.3 单 cell_type 池太小（排除一种设计）

| cell_type | train(site1) 样本 | 每类均值 |
|---|---|---|
| HUVEC | 19,671 | 17.3 |
| RPE | 8,623 | 7.6 |
| HEPG2 | 8,622 | 7.6 |
| U2OS | 3,696 | 3.2 |

→ 「把单一 cell_type 当 InD、其他 cell_type 当 far-OOD」这种 RxRx1 特有设定**样本量不足**（每类 3–17 张），
只能作为降规模消融，不适合作为主实验。

### 1.4 类划分方案与池规模（实测）

| 方案 | InD 类数 | InD 池样本（每类） | D 类数 | D 池样本（每类） |
|---|---|---|---|---|
| 论文式 连续 60%（id 0–682，含控制类） | 683 | 22,515（33.0） | 456 | 18,097（29–153） |
| 连续 40% | 456 | 15,029（33.0） | 683 | 25,583（29–153） |
| 连续 80% | 911 | 30,030（33.0） | 228 | 10,582（29–153） |
| **方案 B：控制类归 InD + treatment 40%** | 474 | 18,697（39.4） | 665 | 21,915（33.0） |
| **方案 B：控制类归 InD + treatment 50%** | 585 | 22,359（38.2） | 554 | 18,253（33.0） |
| **方案 B：控制类归 InD + treatment 60%** | 696 | 26,018（37.4） | 443 | 14,594（33.0） |
| **方案 B：控制类归 InD + treatment 80%** | 917 | 33,304（36.3） | 222 | 7,308（33.0） |

> 方案 B 的 InD 池 = 31 控制类（4,097 张，128–153/类）+ treatment-InD 类（~33/类）。
> 上表均为 `train(site1)` 实测；60% 对应 `665 = round(1108×0.6)` 个 treatment 类。
> 比例选择的完整讨论（为什么是 60%、要不要改成 50%）见
> [`exp1_osal/notes.md`](exp1_osal/notes.md) §2.2.1。

---


---

## 2. 两个实验的关系（为什么两个都要做）

| | 实验一 OSDAL | 实验二 域偏移纯 AL |
|---|---|---|
| 池组成 | I + D + O（三类混合） | 纯 I（多 batch） |
| JODA 组件 | 训练(OE) + **过滤** + 选择 + 平衡 | **仅** 选择 + 平衡 |
| 论文覆盖 | ✅ 核心场景 | ❌ 未覆盖（新设定） |
| RxRx1 难点利用 | 类别多、D 同模态（near-OOD） | **batch effect / 域偏移** |
| 主要产出 | 过滤精度、发现速度、目标类准确率 | 标注效率曲线、gen gap、最差组 |
| 时间 | ~1–2 天/seed（含基线） | ~半天–1 天/seed |

两者互补：实验一验证 JODA 的**完整卖点**；实验二检验其**选择策略在真实域偏移下的可迁移性**——
后者正是 RxRx1 相对 CIFAR 的独特价值，也是论文没有回答的问题。

### 2.1 最终形态：两个实验的**综合改良**（roadmap，非本阶段目标）

> 最终想做的不是"两个平行实验"，而是**在域偏移成立的条件下做 OSDAL**（可记为
> **RxRx1-OSDAL-DS**：Open-Set Discovery AL under Domain Shift）。
> 下面是把这个目标拆开后的四个综合点，也是当前两个实验各自"欠一半"的地方。

| # | 综合点 | 现在的缺口 | 综合后的做法 |
|---|---|---|---|
| S1 | **role 与域的双层切分** | 实验一只按**类**切 I/D/O；实验二只按**实验批次**切 train/val/test，池是纯 I | 把 U 按「类 × 域」二维切：I = 已知类×源域，D = 新类（可跨域发现），O = 无关数据 **+ 域外/无关联批次** |
| S2 | **域偏移下的 OOD 阈值** | 实验一的 energy 阈值在**同分布**上拟合才稳；实验二的过滤被整体跳过 | 阈值改为**按 batch/域分层估计**，或在跨域时在线重估，避免"源域拟合、目标域失效" |
| S3 | **域感知的选择** | SISOMe 的 $d_{in}/d_{out}$ 建立在**源批次的潜空间**上，可能偏向源域；类平衡 Eq.(4) 只平衡**类**，不平衡**域** | 在 Eq.(4) 之外加 **batch/域平衡项**（把 "class share" 替换为 "domain share"），并加域多样性正则 |
| S4 | **统一的评测口径** | 实验一只报 target-class accuracy / precision；实验二只报 OOD acc / gen_gap / worst-group | 统一报告：`acc_target` × `gen_gap` × `worst_celltype` × `selection_precision` × `n_discovered` |

**架构上其实已经为此留好了口子**（不需要推倒重来）：

- `data_config` 已经是一个自由 dict，可以同时声明 `ind_original_class_ids` /
  `discovery_original_class_ids` / `far_data_config`，再加一份 `domain_split`
  （如 "按 experiment 划分源域/目标域"）即可；
- `AlteredDataset.map` 是**纯加法映射**（InD→`0..C-1`、D→`C..`、O→`-1`），
  未来把它扩成 "class-map × domain-map" 的直积即可，其余链路（`OpenDataSetHandler`、
  `JodaLogic`、指标）不用动；
- `sample_selection.filter_osr` 与 `JodaLogic.apply_class_balance` 都是**可替换的独立函数**，
  是注入"域平衡/域鲁棒阈值"的最小侵入点。

**建议路线**：

1. **阶段 1（现在）**：实验一 —— 先把论文的 OSDAL 在 RxRx1 上复现 + 适配（[`exp1_osal/protocol.md`](exp1_osal/protocol.md)）。
2. **阶段 2**：实验二 —— 论文之外的"选择策略在域偏移下的可迁移性"压力测试（[`exp2_domain_shift/design.md`](exp2_domain_shift/design.md)）。
3. **阶段 3**：综合 —— 用阶段 1 的 loader/role 机制 + 阶段 2 的域切分，做 RxRx1-OSDAL-DS（[`exp3_combined/design.md`](exp3_combined/design.md)），
   并实现 S2/S3 两个改良点（这正是可以写成方法贡献的地方）。

---

## 3. 需要的代码改动（总表）

> 下表为两实验共用的落地清单。其中 §2.x 指 [`exp1_osal/notes.md`](exp1_osal/notes.md) 的对应小节，
> §3.x 指 [`exp2_domain_shift/design.md`](exp2_domain_shift/design.md)。

| # | 文件 | 改动 | 类型 |
|---|---|---|---|
| 1 | `data_loaders/classification/rxrx1_data_loader.py` | 新增 `RxRx1ImageListDataset` + `load_rxrx1()`（§2.9） | 🆕 新写 |
| 2 | `load_classification_dataset.py` | ① 顶部 import ② `dataset_factory` 注册 `"rxrx1"` ③ train/eval `trans_map` 加 `"rxrx1-ta"` ④ 增加 rxrx1 的 eval-wrap 分支（方案 T2） | ✏️ 4 处小改 |
| 3 | `load_classification_dataset.py` | 新增 `Rotate90` / `InstanceStandardize` transform 类（§2.11） | 🆕 新写 |
| 4 | `dataset_registry.py` | `ALL_DATASETS` 追加 `"rxrx1"`, `"rxrx1-ta"`, `"rxrx1-test"` | ✏️ 1 行 |
| 5 | `run_configs/rxrx1_*.yml` | 冒烟 + 主实验（方案 A/B）+ 消融 | 🆕 新写 |
| 6 | `scripts/build_rxrx1_memmap.py` + 预转缓存 | 24.7 GB memmap，I/O 提速 ~450×（§2.14） | 🆕 建议 |
| 7 | `data/wilds/rxrx1_v1.0` | ✅ 软链接已建（被 `.gitignore` 忽略） | ✅ 完成 |
| 8 | `ood_detection_datasets.py` | 仅当采用 F3（stock `far_dataset`）时需要加 `rxrx1` 预处理条目；F1/F2 不需要 | ⭕ 可选 |

> 不需要改动的部分（**全部复用**）：`AlteredDataset`、`apply_open_set_modifications`、
> `OpenDataSetHandler`、`JodaLogic`/`JodaQuery`、`OutlierExposure`、`ClassificationHandler` 指标。

---

## 4. 待你拍板的设计决策

| # | 决策点 | 我的建议 |
|---|---|---|
| D1 | InD 类划分：论文式连续 60%（id 0–682）还是「控制类归 InD + treatment 若干 %」；比例取 60% 还是 50% | **方案 B 为主**（控制类归 InD）；比例 **60% 与 50% 并列**。子决策 B1–B4 见 [`exp1_osal/notes.md`](exp1_osal/notes.md) §2.2.1；正式协议见 [`exp1_osal/protocol.md`](exp1_osal/protocol.md) §3.2 / §7 |
| D2 | 远 OOD 来源 | 主：`noise` + `mnist`（可比、易得）；难：自备 `foreign-micro`（建议） |
| D3 | 初始 labeled 预算 | 5 张/InD 类（≈3.4k）；消融 2/10 |
| D4 | $t_e$ | 5（消融 3/10） |
| D5 | 每轮 query | 1000（实验一）/ 1139（实验二） |
| D6 | 训练 epoch/轮 | 预算优先 100；论文可比时 200 |
| D7 | 实验二是否补 `id_test` 离线指标 | **建议补**（分离「发现」与「域偏移」） |

---

## 附录 A：RxRx1 数字速查（本文件实测）

```
总样本 125,510 | 类 1,139 (1108 treat + 30 pos-ctrl + 1 neg-ctrl) | 实验 51 | cell_type 4

train(site1)  40,612  每类 min29 / mean35.7 / median33 / max153   ← AL 池
id_test(site2)40,612  同 33 实验, site2                          ← 同分布测试
val            9,854  4 个 OOD 实验                              ← 选模型
test          34,432  14 个 OOD 实验                             ← 主测试
实验零重叠: train∩val = train∩test = ∅

InD 池(连续60%,683类)=22,515(33/类)   D 池=18,097(29-153/类)
InD 池(控制类+60%,696类)=26,018(37.4/类) D 池=14,594(33/类)   ← 方案 B 主设定
InD 池(控制类+50%,585类)=22,359(38.2/类) D 池=18,253(33/类)
单 cell_type 池: HUVEC 19,671(17.3/类) RPE 8,623(7.6) HEPG2 8,622(7.6) U2OS 3,696(3.2)
```

## 附录 B：数据软链接

```bash
# 已创建
data/wilds/rxrx1_v1.0 -> /data/yuxuan/rxrx1_v1.0     # .gitignore 已忽略 data
```
