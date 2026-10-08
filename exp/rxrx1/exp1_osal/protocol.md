# 实验协议 · 实验一：RxRx1 上的 JODA 开放式发现主动学习（OSDAL）

| 项 | 说明 |
|---|---|
| 版本 | v0.1（草稿，可开始实施） |
| 最近更新 | 2026-10-04 |
| 状态 | 设计完成，待实现 |
| 配套文件 | 全量记录 / 实现细节 / 坑点 / 可复用清单 → [`notes.md`](notes.md)；共用背景 → [`../README.md`](../README.md) |

---

## 1. 实验背景

**数据集（RxRx1，WILDS 版）**：125,510 张 256×256×3 荧光显微图像；**1,139 类** siRNA
（1,108 treatment + 30 positive control + 1 negative control/EMPTY）；51 个 experiment（批次）；
4 种 cell_type（HUVEC / RPE / HEPG2 / U2OS）。任务是由细胞**形态学变化**反推 siRNA，
不同 experiment 之间存在显著 **batch effect**。

**划分**（`train` 与 `val`/`test` 的 experiment **零重叠**；1,139 类在四个划分中完全共享）：

| split | 样本 | experiment | 角色 |
|---|---|---|---|
| `train`（site 1） | 40,612 | 33 | AL 池 |
| `id_test`（`train` 且 site 2） | 40,612 | 33（同 train） | 同分布测试 |
| `val` | 9,854 | 4（留出） | 早停 / 选模型 |
| `test` | 34,432 | 14（留出） | 主测试 |

**方法（JODA，Schmidt et al. 2025）**：在 OSDAL 场景下，未标注池同时含
已知类 **I**、待发现类 **D**（near-OOD）、无关数据 **O**（far-OOD）。
JODA 三阶段：① OE 损失训练 → ② energy + Youden 阈值过滤 O → ③ SISOMe + 类平衡（Eq.4）选样。

**本实验相对论文的差异**（决定了不能照搬论文超参）：

| 维度 | 论文（CIFAR-100 等） | 本实验（RxRx1） |
|---|---|---|
| 类别数 | 10 / 100 / 200 | **1,139** |
| 每类池内样本 | 500 | **~33**（treatment） |
| 域偏移 | 无（同分布 test） | **有**（test 是留出实验） |
| far-OOD | 现成数据集（MNIST / Places365…） | **需自行构造** |
| 主干 | ResNet18 **从零训** 200 epoch | 必须 **ImageNet 预训练**（33 张/类无法从零训） |

---

## 2. 实验目的与假设

**H1｜联合过滤机制在"类多、样本稀疏、需自造 OOD"的真实场景下是否稳定优越。**
在 RxRx1 上验证 JODA 的过滤 + 发现联合机制，是否仍能维持高选择精度与最快的类发现速度。
→ 检验量：`selection_precision`、`n_discovered`、`acc_target`，对照 Random / Ent / LLoss / Badge / LfOSA。

**H2｜新类发现机制在域偏移下是否可靠。**
本实验的评测集是**留出实验**，即发现能力必须在域偏移下成立。检验 JODA 能否同时做好
"发现新类"与"泛化到新批次"。
→ 检验量：`acc_target` 在 `id_test` 与 `test` 上的差距（`gen_gap`）。

**H3｜新类选取机制是否需要优化（是否有比固定 $t_e$ 更好的提升策略）。**
已知信息（前人实验）提示选取 / 提升机制可能过于简单；在"每类池内仅 ~33 张"的 RxRx1 上，
固定 $t_e$ 尤其可疑（$t_e$ 太大 → 永不提升；太小 → 只凭极少样本就认定新类）。
→ 假设：动态 $t_e$（随 cycle、类样本分布、选择精度自适应）优于固定 $t_e$。
→ 检验量：固定 $t_e \in \{3, 5, 10\}$ vs 动态策略，比较 `n_discovered` / `acc_target` 曲线与最终值。

---

## 3. 实验设计

### 3.1 变量声明

**表 1 · 类别与池变量**

| 变量 | 符号 | 取值 | 来源 / 说明 |
|---|---|---|---|
| InD 类别数 | $I$ | **696** | $\mathrm{round}(1108 \times 60\%) + 31$；60% 取自论文[^1]，31 为控制类数 |
| Discovery 类别数 | $D$ | **443** | $1139 - 696$ |
| AL 池 / 未标注池大小 | $U$ | **40,612** | `train`(site 1) |
| 初始标注池大小 | $L_0$ | **3,480** | $5 \times 696$（5 张 / InD 类） |

**表 2 · 迭代与训练变量**

| 变量 | 符号 | 取值 | 说明 |
|---|---|---|---|
| 每轮查询量 | $q$ | **1,000** | ≈1.4 张 / InD 类 / 轮 |
| 迭代轮数 | $T$ | **9** | cycle 0（初始）+ 8 次查询 |
| 累计标注量 | — | 11,480（池的 28.3%） | $L_0 + 8q$；对齐论文时用 ~20,300（50%） |
| 新类提升阈值 | $t_e$ | **5**（消融 3 / 10） | 每类池样本(33)的 15%；论文 10% 等比例 ≈ 3.3 |
| 训练策略 | — | `retrain` | 每轮从零重训（AL 公平比较标准） |
| 主干 / 输入 | — | ResNet18(torchvision) / 256×256 | ImageNet 预训练；逐图实例标准化 |
| 优化器 | — | Adam，lr $10^{-3}$，wd $10^{-5}$ | 对齐论文 |
| 随机种子 | — | 0 / 1 / 2 | AL 方差大，≥3 seeds |

### 3.2 类别角色划分（InD / Discovery）

| 方案 | InD 构成 | InD 类数 | D 类数 | 说明 |
|---|---|---|---|---|
| **B（主设定）** | 31 控制类 + treatment 前 665 类 | **696** | 443 | 控制类是天然的先验已知参照；EMPTY 不应被当作"待发现的新类" |
| A（论文式，对照） | 按 `sirna_id` 连续取前 60%（0–682） | 683 | 456 | 与论文口径直接可比 |

- **InD 比例**：以 **60%** 为主（对齐论文）；**另设 50% 为并列主设定**——RxRx1 每类仅 33 张，
  60% 时 InD 初期的 696 类 × 5 张过于吃紧。50% 属对论文的**有意偏离**，结论中需单独标注。
  消融：40% / 50% / 60% / 80%。
- **treatment 类的挑选方式**：按 `sirna_id` 连续（论文惯例）为主；固定 seed 随机作为稳健性检查。
- **控制类不降采样**（保留数据集自身 4× 过采样），以便检验 Eq.(4) 类平衡项的作用。
- 类别划分以 **json 固化**（`configs/splits/*.json`），保证可复现且与 loader 解耦。

### 3.3 远 OOD（far-OOD）的构造

RxRx1 本身不含远 OOD，需构造；**关键是让 O 与 I/D 走同一套变换与归一化**（否则 energy 假性可分）。

| 编号 | 来源 | 用途 |
|---|---|---|
| **F1（主）** | 外部图像目录（MNIST→PNG 等），由 loader 直接追加进池并打标 $-1$ | 主实验 |
| F2 | 随机噪声（`randn`），无需数据 | 链路冒烟 |
| F3 | 复用 stock 分支（`far_dataset: mnist/places365F`） | 仅作交叉验证（归一化口径不同） |

- InD : O 比例：以 **1:1** 为主，消融 5:1 与 1:2（对应论文 Fig. 8）。
- 指定 O 的样本量时需扣除「初始标注池中自动混入的少量 O」——JODA 需要 labeled 池中同时存在
  InD 与 far-OOD 才能估计过滤阈值；cycle 0 只有 InD，过滤会被自动跳过（与论文一致）。

### 3.4 主动学习迭代协议

```
cycle 0 : 从 L0（5 张 / InD 类）训练 → 评测 → 选择 q 个样本标注
cycle i : 标注集 += q；若有 D 类累计标注 ≥ t_e 则提升为 InD（头部扩容）
          → 重新训练 → 评测
...
cycle 8 : 累计标注 11,480 张
```

- 每轮结束后：过滤（energy + Youden 阈值，基于 labeled 池中的 InD 与 O）→ SISOMe 排序 → 取 top-$q$。
- **模型每轮重训**（`retrain`），输出头随 InD 类数增长。

### 3.5 训练配置

| 项 | 取值 |
|---|---|
| 模型 | ResNet18（torchvision，stride-2 stem + maxpool），ImageNet 预训练 |
| 输入 | 256×256；**逐图实例标准化** `x = (x − mean(x)) / (std(x) + ε)`（按通道） |
| 训练增强 | 随机 90° 旋转 + 水平翻转（均在标准化之前 / 之后等价——二者都是像素置换） |
| 损失 | `OutlierExposure`（CE on InD + $\lambda_{OE}$ · OE on label < 0），$\lambda_{OE} = 0.5$ |
| batch size | 64 |
| epochs / cycle | 100（预算充足时对齐论文的 200） |
| 早停 | `val_acc` |
| 数据加载 | `num_workers=8`（PNG 解码是瓶颈；建议先建 memmap 缓存） |

### 3.6 对照组与消融

| 编号 | 设置 | 目的 |
|---|---|---|
| E1-0 | JODA（主） | — |
| E1-1 | Random / Ent / LLoss / Badge / LfOSA | 与论文 Fig. 3 对齐的方法对照 |
| E1-2 | JODA **No OE**（`loss_func: CrossEntropy`） | 验证 OE 损失 |
| E1-3 | JODA **No Filter**（`apply_ood_filter: false`） | 验证过滤 |
| E1-4 | JODA **No Balancing**（`acl_balancing: false`） | 验证 Eq.(4) 类平衡 |
| E1-5 | $\lambda_{OE} \in \{0.25, 0.5, 1.0\}$ | 超参鲁棒性（论文 Fig. 5） |
| E1-6 | InD 比例 40 / 50 / 60 / 80% | 论文 Fig. 7 对应 |
| E1-7 | O ∈ {F1, F2, F3}、InD:O ∈ {5:1, 1:1, 1:2} | 论文 Fig. 3 / Fig. 8 对应 |
| E1-8 | $t_e \in \{3, 5, 10\}$ × {固定, **动态**} | **H3** |
| E1-9 | 预训练 on/off；`initial_per_class` ∈ {2, 5, 10} | 小样本敏感性（RxRx1 特有） |

### 3.7 运行命名

```
exp/rxrx1/exp1_osal/results/<method>_<scheme><ratio>_<ood>_te<te>_r<seed>/
例：joda_B60_mnist_te5_r0/     random_B60_mnist_te5_r0/
```

---

## 4. 评价指标

### 4.1 主指标

| 指标 | 定义 | 论文对应 |
|---|---|---|
| `acc_target` | 目标类 $C+K$ 上的准确率，**未发现类按 0 计** | 论文主指标 (a) |
| `selection_precision` | 每轮被选中样本中 (I+D) 的占比，$(I+D)/(I+D+O)$ | 论文 (b) |
| `n_discovered` | 被提升为 InD 的 D 类数量 | 论文 (c) |

### 4.2 副指标

| 指标 | 定义 | 用途 |
|---|---|---|
| `acc_inD` | InD 类上的准确率（代码 `test_acc`） | 分离"模型质量"与"缩放因子" |
| `selection_recall` | 已标注 InD / 池内 InD | 覆盖率 |
| `gen_gap` | `acc_target(id_test) − acc_target(test)` | **H2**：域偏移代价 |
| `acc_worst_celltype` / `acc_avg_celltype` | 按 cell_type 分组的最差 / 平均组准确率 | WILDS 标准；U2OS 是天然短板 |
| `query_time` | 每轮选择耗时 | 方法效率 |
| `n_labeled` | 累计标注量 | 曲线横轴 |

### 4.3 指标口径注意事项

1. **`acc_target` 带缩放**：代码实现为 $C_t/(C+K) \times \mathrm{bal\_acc}(\mathrm{InD})$，
   因此 cycle 0 的**上限**就是 $C/(C+K)$（60% 方案为 **0.611**）。
   **不同 InD 比例的曲线不可直接比较**；看曲线时请同时看 `acc_inD`。
2. **`avg_acc` 偏乐观**：$t_e = 5$ 意味着"被发现"的类只有 5 张标注，其真实测试精度不会好，
   而论文口径假设"发现即可正确分类"。→ 需补一个**严格版副指标**：仅在已发现类上的真实准确率。
3. **`acc_target` 与 `selection_precision` 必须一起看**：宁可"漏选"也不能"选错"
   （漏选只损失效率，选错会引入污染并误导发现）。

---

## 5. 实验矩阵（主实验）

| 维度 | 取值 | 组合数 |
|---|---|---|
| 方法 | JODA (+5 个对照) | 6 |
| 类别方案 | B60、B50 | 2 |
| 远 OOD | F1(mnist)、F2(random) | 2 |
| seeds | 0, 1, 2 | 3 |
| **主实验合计** | — | **72 runs** |

消融（E1-2 … E1-9）按需追加，统一复用同一份数据缓存与同一批划分 json。

---

## 6. 实施前提

- 环境：conda env `joda`（torch 1.13.1）；数据：`data/wilds/rxrx1_v1.0`（软链接已建）
- 代码接入：RxRx1 loader + 变换 + 4 处注册（方案见 [`notes.md`](notes.md) §2.7–2.13）
- 冒烟：`F2(random)` + `initial_per_class=2` + `query_size=200` + `cycles=2` + `num_epochs=1` 全链路跑通

> 具体的执行步骤、运行日志、踩坑记录在实验开始后记入 [`notes.md`](notes.md) 与本目录 `results/`。

---

## 7. 待定决策

| # | 决策点 | 建议 |
|---|---|---|
| B1 | 主比例：60% / 50% / 两者并列 | **两者并列**，50% 标注为 RxRx1 特有偏离 |
| B2 | treatment 类挑选：连续 / 随机 | 连续为主，随机作稳健性 |
| B3 | 控制类是否降采样 | 不降采样（保留真实不平衡） |
| B4 | 划分落地：现算 / json | **json** |
| B5 | 累计标注量：28%（现设计）/ 50%（对齐论文） | **50%**（`q=2000` 或增轮数） |
| B6 | 预训练：on（必须） | on；"从零训"降级为消融 |
| B7 | 是否新增"严格版 `acc_target`"副指标 | **是** |
| B8 | 动态 $t_e$ 的具体规则 | 待定（H3，见 [`notes.md`](notes.md) 的口头设计） |

[^1]: 论文附录 A：*"we initially designated 60% of the dataset classes as InD data, while the
remaining classes were considered discoverable"*。方案 B 把该比例作用在 1,108 个 treatment 类上，
再把 31 个控制类固定放入 InD。
