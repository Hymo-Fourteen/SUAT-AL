# 实验一 · AI 维护区（全量记录 / 坑点 / 可复用清单 / 口头设计）

> ⚠️ **本文件是「信息不丢失」文档（给 AI 与后续接手者），不是规范实验协议。**
>
> - 规范实验协议（背景 / 目的与假设 / 设计 / 指标）→ [`protocol.md`](protocol.md)
> - 本文件保留：全量探索事实、踩过的坑、实现方案与代码级细节、本仓库可复用清单、
>   预算估算、端到端步骤与验收断言、以及尚未落地的口头设计。
> - 共用背景见 [`../README.md`](../README.md)；实验二见 [`../exp2_domain_shift/design.md`](../exp2_domain_shift/design.md)。
>
> 下文小节编号沿用重建前的 `§2.x`，仅为与 `../README.md` 的交叉引用保持一致。



## 2.1 要回答的问题

1. JODA 的 **联合过滤 + 发现** 在「类别数 1139、且 near-OOD 与 InD 同模态」的显微图像上是否仍成立？
2. 在 **远 OOD 从自然的自然图像换成显微图像时**，过滤精度与发现速度如何变化？
3. RxRx1 池远小于 CIFAR 时，JODA 的 warmup / $t_e$ / 类平衡项是否仍稳健？或该如何优化？

## 2.2 协议设计


| 项目            | 设计                                                                                                            | 理由                                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 池 U            | `train`(site1) 40,612 张（全部 cell_type）                                                                      | 每类 33 张，是目前唯一规模够用的池                                                                                                   |
| InD 类 I        | 方案 A：id 0–682（683 类）；**方案 B（主）**：31 控制类 + **665** treatment = **696** 类（= round(1108×60%)） | A 与论文可直接对比；B 更符合生物学（控制类天然是先验已知的参照，EMPTY 不该被当作「待发现的新类」）。**比例选择的完整讨论见 §2.2.1** |
| 待发现类 D      | 其余类（自动）                                                                                                  | near-OOD，与 I 同模态 —— 与论文设定完全对应                                                                                        |
| 远 OOD O        | 见 §2.3 菜单                                                                                                   | far-OOD                                                                                                                              |
| 初始 labeled 池 | **5 张/InD 类**（方案 A：3,415；方案 B：**3,480**）                                                             | 论文 2000/60 类 ≈ 33/类，但 RxRx1 只有 33/类，故取 5/类，留出迭代空间（见 §2.2.1）                                                 |
| query q         | **1000 / 轮**（≈1.5 张/InD 类/轮）                                                                             | 池仅 4 万，论文的 2500 占比过大                                                                                                      |
| cycles          | **9**（cycle 0 初始 + 8 次查询）                                                                                | 最终 labeled ≈ 11.4k ≈ 池的 28%                                                                                                    |
| $t_e$           | **5**（消融 3 / 5 / 10）                                                                                        | 🔴 必须远小于论文的 50：D 类池内只有 ~33 张                                                                                          |
| 训练            | `strategy: retrain`（每轮从零重训）；ResNet18/`Resnet18T` @256；Adam 1e-3；实例标准化                           | 论文 200 epoch/轮；预算允许时对齐 200，否则 50–100                                                                                  |
| 评测            | val(4 OOD 实验) 选模型；**test(14 OOD 实验)** 报主指标；另报 **id_test** 分离「发现」与「域偏移」               | 论文是同分布 test；RxRx1 test 是留出实验，同时报 id_test 可拆分两种难度                                                              |

## 2.2.1 InD 比例为什么是 60%？—— 一个需要明确拍板的选择

**先回答字面问题**：方案 B 里的 `664`（正确四舍五入应为 **665**）来自
`1108 个 treatment × 60%`，而 60% 是**论文的默认 InD 比例**（论文附录 A：
"we initially designated **60% of the dataset classes as InD** data, while the remaining
classes were considered discoverable"）。**因为用的是 60% 而不是 50%，所以它超过了半数**；
再加上 31 个控制类固定进 InD，最终 InD = **696 类 = 全部 1139 类的 61.1%**。

池的精确构成（`train(site1)` 实测）：


| 成分                    | 类数  | 池内样本 | 每类                   |
| ----------------------- | ----- | -------- | ---------------------- |
| 31 控制类（固定进 InD） | 31    | 4,097    | 128–153（均值 132.2） |
| treatment（待按比例切） | 1,108 | 36,515   | 29–33（均值 33.0）    |
| 合计                    | 1,139 | 40,612   | —                     |

**不同 InD 比例下的规模（方案 B）**


| treatment InD 比例 | InD treatment | InD 类数 C | D 类数 K | InD 池     | D 池       | InD 类别占比 | 指标缩放$C/(C+K)$ |
| ------------------ | ------------- | ---------- | -------- | ---------- | ---------- | ------------ | ----------------- |
| 40%                | 443           | 474        | 665      | 18,697     | 21,915     | 41.6%        | 0.416             |
| **50%**            | 554           | **585**    | 554      | 22,359     | 18,253     | 51.4%        | 0.514             |
| **60%（论文）**    | **665**       | **696**    | **443**  | **26,018** | **14,594** | **61.1%**    | **0.611**         |
| 70%                | 776           | 807        | 332      | 29,678     | 10,934     | 70.9%        | 0.709             |
| 80%                | 886           | 917        | 222      | 33,304     | 7,308      | 80.5%        | 0.805             |

**一个关键观察：这个比例并不决定「D 类能否被发现」**

因为 treatment 类每类样本几乎恒定（~33），D 池大小 $\propto K$，所以在均匀选择下
**每个 D 类被抽到的期望样本数恒为**
$(q\times\text{cycles})\times 33/\text{len(pool)} = 8000\times 33/40{,}612 \approx 6.5$，**与比例无关**。
也就是说：只要 $t_e \le 5$，**任何比例下所有 D 类在理论上都可被提升**。

所以比例真正控制的是这四件事：

1. **InD 训练集规模**（→ 模型强度）：60% → 26,018 张；50% → 22,359 张；
2. **发现任务的规模与曲线分辨率**：D 类数 665（40%）→ 222（80%）；
3. **论文指标的缩放因子** $C/(C+K)$：注意 `avg_acc = C/(C+K) × bal_acc(InD)`，
   cycle 0 就带上了 0.611（60%）的系统性缩放。**若两条曲线用了不同比例，曲线不可直接比较**；
4. **选择器的难度**：D 在池中的占比（54% → 18%）越低，越容易被 InD 淹没。

**建议**

- **主设定：60%** —— 与论文可比，且 §2.5 的 40/60/80 消融正好对齐论文 Fig. 7；
- **并列主设定：50%** —— 理由：RxRx1 每类只有 33 张，60% 时 InD 初期是 696 类 × 5 张 = 3,480 张，
  非常吃紧；50% 把 D 提到 554 类，发现曲线更有区分度，且 $C/(C+K)=0.514$ 更接近「均衡 OSDAL」。
  论文没有 50% 这一点，属于针对 RxRx1 的**有意偏离**，必须在结果里单独标注；
- **消融**：40% / 50% / 60% / 80%（严格对齐论文时用 40/60/80）。

**待定子决策（需拍板）**


| #      | 子决策                 | 选项                                                                   | 建议                                                                         |
| ------ | ---------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **B1** | 主比例                 | 60% / 50% / 两者并列                                                   | **两者并列**；50% 作为 RxRx1 主设定并注明是对论文的偏离                      |
| **B2** | treatment 类的挑选方式 | ① 按`sirna_id` 连续（论文惯例）② 固定 seed 随机                      | **① 连续**（可比）；② 作为稳健性检查                                       |
| **B3** | 控制类是否降采样       | ① 全量（4,097，4× 过采样）② 截到 ~33/类（1,023，与 treatment 对齐） | **① 全量**（保留数据集自身的不平衡，让 Eq.(4) 有真实用武之地）；② 作为消融 |
| **B4** | 类别划分的落地形式     | 运行时现算 / 存成 json                                                 | **存 json**（`exp1_osal/configs/splits/*.json`），保证可复现且与 loader 解耦 |

> 附带一个解释性提醒：由于 `avg_acc` 带 $C/(C+K)$ 缩放，**cycle 0 的 `avg_acc` 上限**
> 就是 $C/(C+K)$（InD 全对、D 全 0）。看曲线时请同时看 `test_acc`（= InD 上的准确率，
> 已由代码单独输出），否则容易把「缩放因子的变化」误读成「模型变好/变差」。

## 2.3 远 OOD 菜单（**关键设计决策**）


| 编号                          | 来源                                                | 下载   | 难度                       | 用途                      |
| ----------------------------- | --------------------------------------------------- | ------ | -------------------------- | ------------------------- |
| O-noise                       | `randn`（`far_dataset: random`）                    | 不需要 | 极容易                     | 冒烟 / sanity，确保链路通 |
| O-mnist                       | MNIST（`far_dataset: mnist`，TorchVision 自动下载） | 小     | 容易                       | 论文 Fig.3 的中档对照     |
| O-places                      | Places365 / ImageNetC-800                           | 大     | 中（自然图像，视觉差异大） | 与论文直接对照            |
| **O-foreign-micro**（建议补） | 其他细胞系/其他显微成像模态的图（视作任务无关）     | 需自备 | **难**（同为显微图像）     | 真正能区分方法的 far-OOD  |

> ⚠️ **重要**：荧光显微图 vs 自然图像（MNIST/Places365）在像素统计上差异极大，
> 过滤会过于容易 —— 论文自己在 TinyImageNet+Places365 也承认接近 OSDAL 的「边界」。
> 若只用 MNIST/Places365，很可能**所有方法的 selection precision 都 ≈1.0，无法区分**。
> 因此建议：**主实验用 O-noise + O-mnist 跑通并给出论文可比点，同时追加 O-foreign-micro 作为难点**。

> ⚠️ **归一化必须一致**：代码里 far-OOD 走 `get_default_preprocessor(base_dataset)` 的**固定 mean/std**，
> 而 RxRx1 主分支必须用 **逐图实例标准化**（见探索结论）。若两者不一致，
> energy 分数会因预处理差异而「假性可分」，过滤指标失去意义。
> **必须让 far-OOD 也走同一套实例标准化**（需在 `default_preprocessing_dict` 增加 `rxrx1` 条目或
> 为 RxRx1 增加专用 OOD 分支）。

## 2.4 指标


| 指标                     | 定义                                     | 论文对应           |
| ------------------------ | ---------------------------------------- | ------------------ |
| `acc_target` ⭐          | 目标类 C+K 上的准确率（未发现类按 0 计） | 论文主指标；(a)    |
| `selection_precision` ⭐ | (I+D)/(I+D+O)，按每个 query 批次统计     | 论文 (b)           |
| `selection_recall`       | 已标注 InD / 池内 InD                    | 代码额外提供       |
| `n_discovered` ⭐        | 被提升的 D 类数（`len(ind_classes)-C`）  | 论文 (c)           |
| `acc_idtest`             | 同分布 id_test 上（分离域偏移）          | 论文无，RxRx1 特有 |
| `acc_worst_celltype`     | cell_type 最差组                         | WILDS 标准         |

## 2.5 对照与消融


| 编号     | 对照                                           | 目的                                  |
| -------- | ---------------------------------------------- | ------------------------------------- |
| **E1-0** | JODA（主）                                     | —                                    |
| E1-1     | Joda**No OE**（`loss_func: CrossEntropy`）     | 验证 OE 损失的作用（论文 Fig.5 对应） |
| E1-2     | Joda**No Filter**（`apply_ood_filter: false`） | 验证过滤的作用                        |
| E1-3     | Joda**No balancing**（`acl_balancing: false`） | 验证 Eq.(4)                           |
| E1-4     | `lambda_oe` 0.25 / 1.0                         | 论文 Fig.5 超参鲁棒性                 |
| E1-5     | 初始类比例 40/60/80%                           | 论文 Fig.7 对应                       |
| E1-6     | O = noise / mnist / places / foreign-micro     | 论文 Fig.3 对应 + RxRx1 特有难点      |
| E1-7     | Random / Ent / LLoss / Badge / LfOSA           | 论文 Fig.3 的对照方法                 |

> 原仓库已自带消融配置：`JODA/config_standalone/ablation_no_oe*.json`、`ablation_no_filter*.json`、
> `ablation_lambda*.json`（来自另一方的修改），可直接改造。

## 2.6 配置草稿（方案 B + O-mnist）

```yaml
experiment_config:
  method_type: Joda
  query_scenario: Pool
  data_scenario: osal-extending
  dataset: rxrx1
  dataset_path: ./data/wilds
  static_configuration: ta
  ind_classes: [0, 1, ..., 694]  # ★ 必须是重编号后的 list(range(C))；真实 sirna_id 放 data_config
  near_classes: []               # 留空：apply_open_set_modifications 自动填 [C .. C+K-1]
  far_classes: [-1]              # ★ 远 OOD 的标签（AlteredDataset 需要它才会映射到 -1）
  far_dataset: "None"            # ★ 远 OOD 由 loader 内部构造，不走 stock OOD 分支（见 §2.12）
  expanding_th: 5                # ★ 不是论文的 50（池内每类只有 ~33 张）
  opensetmode: All
  cycles: 9
  query_size: 1000
  over_selection: 2.0
  is_percent: false
  task: classification
  engine: pytorch
  seed: 0
  data_config:
    image_root: ./data/wilds/rxrx1_v1.0     # 含 metadata.csv 与 images/
    pool_split: train_site1                  # WILDS: dataset==train & site==1
    ind_original_class_ids: [ ... ]          # 真实 sirna_id：31 控制类(1108-1138) + treatment 0..664
    discovery_original_class_ids: [ ... ]    # 真实 sirna_id：treatment 665..1107
    initial_labeled_per_class: 5
    image_size: 256
    far_data_config:                         # 可选：外部远 OOD 图像目录
      image_root: ./data/wilds/ood/<name>
      max_samples_per_class: 200
training_config:
  loss_func: OutlierExposure
  lambda_oe: 0.5
  triplet_mode: triplet     # ★ 必须：让 OE 用 label<0 的远 OOD（off 会用 >=num_classes 的近 OOD，方向相反）
  model: Resnet18T          # 256×256 必须用 stride-2 stem 版本
  pretrained: true
  num_epochs: 100           # 预算充足时对齐论文的 200
  batch_size: 64
  optimizer: Adam
  lr: 0.001
  wdcay: 0.00001
  num_workers: 8
  coverage_al: incremental
  surprise_strategy: surprise
  gain_mode: energy
  seperator: metric
  sep_metric: energy
  apply_ood_filter: true
  acl_balancing: true
  sigmoids: { "AdaptiveAvgPool2d-1": 100, "Sequential-3": 1000, "Sequential-2": 0.001, "Sequential-1": 0.001 }
```

## 2.7 实现路径总览：**实验一 ≈ 把 `plankton_dataloader.py` 的范式套到 RxRx1 上**

> 🎯 关键发现：合并进来的 fork-B 代码 `data_loaders/classification/plankton_dataloader.py`
> **已经实现了一个完整的、不依赖论文式「连续类 id」约定的 OSDAL 池构造**：
> 它按显式原始类 id 建池、自己把 far-OOD 追加到池尾并打标 `-1`、
> 返回的 `ind_classes` 恰好是 `list(range(C))`，从而让后续的
> `apply_open_set_modifications` / `AlteredDataset` 只需做一次「恒等重映射」。
> **RxRx1 要做的核心工作，就是把它的「读 `splits.json` + 拼图片路径」换成
> 「读 `metadata.csv` + 拼 `images/<exp>/Plate<plate>/<well>_s<site>.png`」。**


| 环节                                                   | 复用？      | 复用对象                                                |
| ------------------------------------------------------ | ----------- | ------------------------------------------------------- |
| 池构造（初始 InD / 未标注 I+D / far-OOD 混入并打`-1`） | ✅ 范式复用 | `plankton_dataloader.load_plankton_pool`                |
| InD/D/O 标签重映射 + val/test 过滤 + near 类推导       | ✅ 完全复用 | `apply_open_set_modifications` + `AlteredDataset`       |
| 新类提升（$t_e$）                                      | ✅ 完全复用 | `OpenDataSetHandler.update_data_pool()`                 |
| OOD 过滤（energy+ROC+Youden）+ SISOMe 选择 + 类平衡    | ✅ 完全复用 | `JodaQuery` / `JodaLogic`                               |
| OE 训练损失                                            | ✅ 完全复用 | `OutlierExposure`（`loss_functions.py:305`）            |
| 指标（extending acc / selection precision / recall）   | ✅ 完全复用 | `ClassificationHandler` + `sample_selection.filter_osr` |
| 消融配置模板                                           | ✅ 复用     | `config_standalone/ablation_*.json`                     |
| 元数据解析 + 图像路径 + 划分                           | ❌ 新写     | `rxrx1_data_loader.py`                                  |
| 变换（实例标准化 / 90° 旋转）                         | ❌ 新增     | 两个`trans_map` + 2 个自定义 transform 类               |
| 注册                                                   | ❌ 3 处小改 | factory / train&eval trans_map /`ALL_DATASETS`          |

---

## 2.8 可复用代码清单（已逐处核对）


| #  | 文件 / 符号                                                              | 作用                                                                            | 复用方式                                                      |
| -- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1  | `plankton_dataloader.py::PlanktonImageListDataset`                       | 真实图片 list 数据集：`samples/targets/transform`、`get_class_counts()`         | 抄结构 →`RxRx1ImageListDataset`                              |
| 2  | `plankton_dataloader.py::load_plankton_pool`                             | 显式类 id 建池、far-OOD 追加、`dataset_config` 组装、7 元组返回                 | 抄结构 →`load_rxrx1`                                         |
| 3  | `ood_detection_datasets.py::AlteredDataset`                              | InD→`0..C-1`、D→`C..C+K-1`、O→`-1`；`set_map()` 支持提升后重映射             | **直接复用，零改动**                                          |
| 4  | `load_classification_dataset.py::apply_open_set_modifications`           | 过滤 val/test 到 InD、推导`near_classes`、设 `all_classes`、包 `AlteredDataset` | **直接复用**（前提：loader 已重编号且 `far_dataset: "None"`） |
| 5  | `data_handler.py::OpenDataSetHandler.update_data_pool`                   | 用`expanding_th` 提升新类、更新 map/`num_classes`、重建 val/test                | **直接复用**                                                  |
| 6  | `Joda_logic.py` / `active_coverage_learning.py::JodaQuery`               | 过滤阈值 + SISOMe + Eq.(4)                                                      | **直接复用**（`data_scenario: osal-extending` 即触发）        |
| 7  | `loss_functions.py::OutlierExposure`                                     | OE 损失                                                                         | **直接复用**（需 `triplet_mode: triplet`）                    |
| 8  | `ClassificationHandler`（`avg_acc` 分支）/ `sample_selection.filter_osr` | 论文三指标的现成实现                                                            | **直接复用**                                                  |
| 9  | `config_standalone/ablation_*.json`                                      | No-OE / No-Filter / λ 消融                                                     | 改字段即可                                                    |
| 10 | `scripts/download_rxrx1.py` + 软链接                                     | 数据就位                                                                        | ✅ 已完成                                                     |

**结论：实验一 ~80% 的机制是现成的，真正要写的只有「RxRx1 元数据 → 样本列表」这一段。**

---

## 2.9 需要新写的代码规格：`rxrx1_data_loader.py`

文件：`JODA/src/joda_al/data_loaders/classification/rxrx1_data_loader.py`

#### 2.9.1 图像路径模板（已实测 400/400 命中）

```
{image_root}/images/{experiment}/Plate{plate}/{well}_s{site}.png
# 例：data/wilds/rxrx1_v1.0/images/HUVEC-01/Plate1/M23_s2.png
```

#### 2.9.2 划分规则（`metadata.csv`，125,510 行）


| 角色      | 过滤条件                           | 样本数 |
| --------- | ---------------------------------- | ------ |
| `pool`    | `dataset=="train" & site==1`       | 40,612 |
| `id_test` | `dataset=="train" & site==2`       | 40,612 |
| `val`     | `dataset=="val"`（4 个留出实验）   | 9,854  |
| `test`    | `dataset=="test"`（14 个留出实验） | 34,432 |

> ⚠️ `metadata.csv` 的 `dataset` 列**只有 train/val/test**，没有 `id_test`；
> 必须按 `dataset==train & site==2` 现拆（Random 基线方案 §2.6-坑 6）。

#### 2.9.3 类结构与契约

```python
class RxRx1ImageListDataset(Dataset):
    """把 (path, target) 列表包成 JODA 要求的接口。"""
    def __init__(self, samples):                 # samples: Sequence[Tuple[str, int]]
        self.samples = [(str(p), int(t)) for p, t in samples]
        self.targets = [t for _, t in self.samples]   # ★ JODA 多处按 list 索引
        self.transform = None                        # ★ 不做变换，交给 JODA 的变换链

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        with Image.open(path) as im:
            im = im.convert("RGB")      # ★ 返回 PIL，不是 tensor
        if self.transform is not None:
            im = self.transform(im)
        return im, target

    def get_class_counts(self, indices):         # ★ 契约要求
        v, c = np.unique(np.asarray(self.targets)[indices], return_counts=True)
        return dict(zip(v.tolist(), c.tolist()))
```

#### 2.9.4 `load_rxrx1(dataset, config, order=1)` 逻辑骨架

```python
def load_rxrx1(dataset, config, order=1):
    data_config = config.get("data_config", {})
    root        = Path(data_config.get("image_root", dataset["path"]))
    meta        = _read_metadata(root)                       # 读一次并缓存
    ind_ids     = [int(v) for v in data_config["ind_original_class_ids"]]
    disc_ids    = [int(v) for v in data_config["discovery_original_class_ids"]]
    initial_pc  = int(data_config["initial_labeled_per_class"])

    # 1) 原始 id -> 重编号：InD=0..C-1, D=C..C+K-1
    target_ids = ind_ids + disc_ids
    target_map = {orig: new for new, orig in enumerate(target_ids)}
    # ★ apply_open_set_modifications 依赖这个恒等前提
    assert list(config["ind_classes"]) == list(range(len(ind_ids)))

    # 2) 池：初始标注(每个 InD 类前 initial_pc 张) + 其余目标 + far-OOD(-1)
    pool_df  = _split_frame(meta, "train_site1")
    init_samples, rest_samples = [], []
    for orig in ind_ids:
        paths = _class_paths(root, pool_df, orig)            # 稳定排序后切片
        init_samples += [(p, target_map[orig]) for p in paths[:initial_pc]]
        rest_samples += [(p, target_map[orig]) for p in paths[initial_pc:]]
    for orig in disc_ids:
        rest_samples += [(p, target_map[orig]) for p in _class_paths(root, pool_df, orig)]
    far_samples = _build_far_ood(root, data_config, meta)     # [(path|-1)] 见 §2.12

    training_pool = RxRx1ImageListDataset(init_samples + rest_samples + far_samples)
    label_idx     = list(range(len(init_samples)))
    unlabeled_idx = list(range(len(init_samples), len(init_samples) + len(rest_samples) + len(far_samples)))

    # 3) val/test：只含目标类（InD+D），供 apply_open_set_modifications 再过滤到 InD
    validation_set = RxRx1ImageListDataset(_target_samples(root, meta, "val",  target_ids, target_map))
    test_set       = RxRx1ImageListDataset(_target_samples(root, meta, "test", target_ids, target_map))

    feature_sizes = set_feature_size([64, 32, 16, 8], [64, 32, 16, 8], [64, 32, 16, 8])
    dataset_config = {
        "num_classes": len(target_ids),        # = C+K；随后被改成 C
        "all_classes": len(target_ids),        # ★ osal-extending 的 extending acc 分母
        "feature_sizes": feature_sizes,
        "encoding_dimension": 512,
        "sample_size": (256, 256),
        "ind_original_class_ids": ind_ids,
        "discovery_original_class_ids": disc_ids,
        "far_original_class_ids": far_ids,
        "class_names": [...],
    }
    return (training_pool, label_idx, [unlabeled_idx],
            validation_set, test_set, dataset_config, DatasetScenarioConverter())
```

**边界校验（必须显式报错，不要静默）**


| 校验                                      | 原因                                                     |
| ----------------------------------------- | -------------------------------------------------------- |
| `ind_ids ∩ disc_ids == ∅`               | 角色冲突                                                 |
| 每个 InD 类在池内样本数`>= initial_pc`    | 避免`initial_labeled_per_class` 抽不满                   |
| 每个 D 类在池内样本数`> 0`                | 避免空类                                                 |
| `set(target_ids) ⊆ metadata.sirna_id`    | 防拼错 id                                                |
| 图片文件存在性抽样/全量检查               | 防路径模板错                                             |
| `config["ind_classes"] == list(range(C))` | `apply_open_set_modifications` 的隐含前提                |
| 池内类不均匀度                            | 已知：treatment 31–33，control 131–153，记录到日志即可 |

---

## 2.10 三处注册点（照 plankton 抄）

以现有 `SYKE-*` 为例（`load_classification_dataset.py`）：

```python
# ① 顶部 import（现有第 18 行附近）
from joda_al.data_loaders.classification.rxrx1_data_loader import load_rxrx1

# ② dataset_factory（现有第 120-121 行附近）
dataset_factory = {
    ...
    "SYKE-ZooScan": load_plankton_pool,
    "SYKE-IFCB": load_plankton_pool,
    "rxrx1": load_rxrx1,          # ★ 新增；匹配 dataset_comb_name.startswith(key)
    "GTAVS": load_gtavs,
    ...
}

# ③ get_training_transformations_classification / get_eval_transformations_classification 的 trans_map
#    注意：查找键是 f'{dataset_config["name"]}-{dataset_config["dataset_scenario"]}'
#    _lookup_dataset_transform 用「最长前缀」匹配，所以用 "rxrx1-ta" 可同时覆盖 "rxrx1-ta-ta"
trans_map = { ..., "rxrx1-ta": rxrx1_train_transform }      # train
trans_map = { ..., "rxrx1-ta": rxrx1_eval_transform }       # eval
```

```python
# ④ dataset_registry.py
ALL_DATASETS = [ ..., "rxrx1", "rxrx1-ta", "rxrx1-test" ] + ...
```

> `config.experiment_config["dataset"]` 建议用 `rxrx1-ta`：
> 于是 `dataset["name"]="rxrx1-ta"`、`dataset_scenario="ta"` →
> `dataset_comb_name="rxrx1-ta-ta"`（factory 前缀命中 `rxrx1`）、
> 变换键 `"rxrx1-ta-ta"`（trans_map 前缀命中 `rxrx1-ta`）。与 SYKE 完全一致。

---

## 2.11 变换设计（代码级，**有一个坑必须避开**）

#### 2.11.1 变换施加位置（已核实）


| 位置                   | 代码                                                                                                                     | 施加的变换                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| **池**（含 far-OOD） | `get_train_loader` → `TransformSubset(pool, idx, **get_training_transformations(...))`                                  | `transform`（单参，作用于 img） |
| **池 + val + test**    | `load_classification_dataset` 里的白名单分支 → `TransformDataset(..., **get_eval_transformations_classification(name))` | `transform`（单参）             |
| **val / test**         | `get_val_loader` / `get_test_loader`                                                                                     | **不再施加任何变换**            |

#### 2.11.2 ⚠️ 坑：非白名单数据集的池会被套两次变换

`load_classification_dataset` 中：

```python
if dataset["name"] not in ["cifar10", ..., "cifar100", "svhn"]:
    training_pool = TransformDataset(training_pool, **get_eval_transformations_classification(dataset["name"]))
    validation_set = TransformDataset(validation_set, **get_eval_transformations_classification(dataset["name"]))
    test_set      = TransformDataset(test_set,      **get_eval_transformations_classification(dataset["name"]))
```

即**非白名单数据集的 pool 会先被套「eval 变换」，再在 `get_train_loader` 里套「train 变换」**。
`plankton` 就是靠「train 变换里不放 `ToTensor`」来绕过这个坑的。

两种可行方案：

**方案 T1（零改动，靠算子可交换性）**

- eval 变换（作用于 pool+val+test）：`Compose([ToTensor(), InstanceStandardize()])`
- train 变换（再叠加于 pool）：`Compose([RandomRotate90_tensor(), RandomHorizontalFlip()])`
- ✅ 成立的理由：**90° 旋转与水平翻转都是像素置换**，与「逐图 per-channel 均值/方差标准化」可交换，
  先标准化再旋转 ≡ 先旋转再标准化。所以顺序虽然反了，结果是对的。
- ⚠️ 限制：**train 变换必须是 tensor-safe 且不含 `ToTensor`/`Normalize`**；
  一旦以后想加 `RandomResizedCrop`/`ColorJitter` 这类非置换算子，这个方案就不成立。

**方案 T2（推荐，改 ~6 行，语义干净）**

在 `load_classification_dataset` 里给 rxrx1 单独分支：

```python
if str(dataset["name"]).startswith("rxrx1"):
    # pool 保持原始 PIL：由 get_train_loader 施加训练增强
    validation_set = TransformDataset(validation_set, **get_eval_transformations_classification(dataset["name"]))
    test_set       = TransformDataset(test_set,       **get_eval_transformations_classification(dataset["name"]))
elif dataset["name"] not in ["cifar10", ..., "svhn"]:
    ... 原逻辑 ...
```

于是：

```python
class Rotate90:                      # 作用于 PIL，k ∈ {0,1,2,3} 均匀采样
    def __call__(self, image):
        return image.rotate(90 * random.randint(0, 3), expand=False)

class InstanceStandardize:           # 作用于 CHW tensor
    def __call__(self, x):
        return (x - x.mean(dim=(1, 2), keepdim=True)) / x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)

rxrx1_train_transform = T.Compose([
    Rotate90(), T.RandomHorizontalFlip(), T.ToTensor(), InstanceStandardize(),
])   # 作用于 PIL
rxrx1_eval_transform = T.Compose([
    T.ToTensor(), InstanceStandardize(),
])                                              # 作用于 PIL
```

> ✅ `InstanceStandardize` 是**唯一正确的归一化**（见 §1 / Random 基线方案 §2.4）：
> 图像全局均值仅 ~0.02–0.06，用 ImageNet 归一化会得到几乎全负的输入。

---

## 2.12 far-OOD 实现方案（3 选 1）


| 方案                                         | 做法                                                                                                          | 归一化一致性                                                                                          | 需要下载                                         | 评价                                                             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| **F1 推荐**：loader 内构造（读外部图片目录） | `data_config.far_data_config.image_root` 指向一个 `ImageFolder` 式目录，loader 把它作为 `(path, -1)` 追加进池 | ✅ 与 InD 共用同一变换链                                                                              | 自备（MNIST→PNG、Places365 子集、或显微域外集） | 与 plankton 完全同构；**唯一能保证实例标准化一致的方式**         |
| F2：loader 内构造 random noise               | 在 loader 里用`torch.randn` 生成，或复用 `NoiseDataset`，写成池内 `-1` 样本                                   | ✅（但无意义，噪声无需标准化）                                                                        | 不需要                                           | **冒烟首选**：零依赖、秒级、可先跑通全链路                       |
| F3：走 stock 分支                            | `far_dataset: mnist / places365F / random`                                                                    | ❌ 走`get_default_preprocessor()` 的**固定 mean/std**，与 RxRx1 的实例标准化不一致 → energy 假性可分 | MNIST 小、Places365 大                           | 仅作交叉验证；需先给`default_preprocessing_dict` 加 `rxrx1` 条目 |

**建议执行顺序**：`F2`（冒烟）→ `F1`（主实验，MNIST-as-PNG + 显微域外集）→ `F3`（可选，与论文数值对拍）。

> 📌 顺带解决了一个此前的顾虑：由于 far-OOD 直接进 `training_pool`，
> `AlteredDataset` 把它标成 `-1`，`JodaLogic.estimate_ood_threshold()` 用 `ood_stats==2` 拿到它 ——
> **过滤链路无需任何改动即可工作**；且当初始 labeled 池还没有 far-OOD 时，
> B 版加的守卫 `if torch.unique(ood_stats).numel() < 2: optimal_threshold=None`
> 会自动跳过过滤，正好对应论文「第 0 轮不过滤」。

---

## 2.13 端到端跑通步骤（含验收检查点）


| 阶段 | 动作                                                                                                                  | 验收信号                                                                                                                                                             |
| ---- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0   | 写`rxrx1_data_loader.py`（F2 far-OOD = random）                                                                       | `python -c "import joda_al..."` 可导入                                                                                                                               |
| P0   | 注册 factory / trans_map /`ALL_DATASETS`                                                                              | `ALL_DATASETS` 含 rxrx1；`check_existence("Joda")` 仍 True                                                                                                           |
| P0   | 写`run_configs/rxrx1_osal_smoke.yml`：`cycles: 2`、`query_size: 200`、`num_epochs: 1`、`initial_labeled_per_class: 2` | —                                                                                                                                                                   |
| P1   | 跑 smoke                                                                                                              | 日志出现`Dataset: rxrx1-ta` / `Method type:Joda` / `New classes: [...]` / `Joda candidates after OOD filtering: X/Y`；`stats_file.csv` 有 `test_acc`、`test_avg_acc` |
| P1   | **变换断言**（最容易错的地方）                                                                                        | 取一个 batch 打印`x.mean()`/`x.std()`：应 ≈ 0 / ≈ 1（                                                                                                              |
| P1   | **映射断言**                                                                                                          | 打印`train_pool.targets` 的直方图：应出现 `0..C-1`（InD）、`C..C+K-1`（D）、`-1`（far）三类                                                                          |
| P1   | 池完整性                                                                                                              | `len(train_pool) == 40612 + len(far_samples)`；`len(label_idx) == C*initial_pc`                                                                                      |
| P2   | 单 cycle 指标 sanity                                                                                                  | `test_acc` 显著高于 `1/C`；`test_avg_acc ≈ (C/(C+K))·bal_acc`                                                                                                      |
| P2   | 类提升 sanity                                                                                                         | 第 2 轮起`New classes` 非空（若 `expanding_th` 太大则为空 → 调小）                                                                                                  |
| P3   | 跑完 9 cycle × 1 seed                                                                                                | 曲线：`test_avg_acc` 单调上升；`Selection/Precision` 稳定                                                                                                            |
| P4   | 补 seed 1/2 + 消融                                                                                                    | —                                                                                                                                                                   |

**冒烟配置要点**：远 OOD 用 F2(random)、`initial_labeled_per_class: 2`、`query_size: 200`、
`expanding_th: 2`、`num_epochs: 1`，全程 CPU 也能在几分钟内结束。

---

## 2.14 预算估算（H100，单 seed，9 cycle，q=1000，696 InD 类）


| 项           | 计算                                                                               | 用时                                                     |
| ------------ | ---------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 初始 labeled | 696 × 5 = 3,480                                                                   | —                                                       |
| 每轮训练     | labeled 从 3.5k → 11.5k，bs64 → 55→180 iter/epoch；日志报 r18@256 ≈ 18 ms/iter | 100 epoch/轮 × 9 轮 ≈**27 min**                        |
| 每轮评估     | val(InD 子集) + test(34,432) + id_test(40,612)                                     | 取决于 I/O（见下）                                       |
| **I/O**      | 单进程 PNG 解码 22 img/s；8 workers ≈ 176 img/s；memmap ≈ 10k img/s              | 一次 test 全量评估：22→26 min，8w→3.3 min，memmap→3 s |
| 合计/seed    | memmap +`num_workers=8`                                                            | **≈ 40–60 min**                                        |
| 合计/seed    | 直读 PNG +`num_workers=8`                                                          | **≈ 2–3 h**                                            |

**结论**：先做 memmap 缓存（125,510×256×256×3 = 24.7 GB，一次写入），
否则 I/O 会成为主要成本。

---

## 2.15 风险与对策（实验一）


| 风险                                        | 可能性 | 影响                                          | 对策                                                  |
| ------------------------------------------- | ------ | --------------------------------------------- | ----------------------------------------------------- |
| 用了 ImageNet 归一化而非实例标准化          | 中     | 准确率崩塌，误判「方法无效」                  | §2.13 的 batch mean/std 断言                         |
| 非白名单导致池被套两次变换                  | 高     | 若 train 变换含`ToTensor` 会直接报错          | 采用 T1（不含 ToTensor）或 T2（单独分支）             |
| far-OOD 与 InD 归一化不一致                 | 中     | energy 假性可分、precision 虚高               | 用 F1/F2 让 O 走同一变换链                            |
| `triplet_mode: off` 导致 OE 作用在近 OOD 上 | 中     | 与论文方向相反，D 被压成低 energy → 发现失败 | 配置强制`triplet_mode: triplet`                       |
| `ind_classes` 未重编号                      | 中     | plankton loader 直接`ValueError`              | 真实 id 放`data_config`，`ind_classes=list(range(C))` |
| `expanding_th` 过大                         | 中     | 「发现类数」恒为 0                            | 5（消融 3/10）；看`New classes` 日志                  |
| 1139 类 head + 256×256 显存                | 低     | OOM                                           | 用`Resnet18T`（stride-2 stem + maxpool）              |
| 单 seed 结论                                | 中     | 方差被误读                                    | ≥3 seeds + 阴影带                                    |
| val(4 实验) 与 test(14 实验) 分布差异       | 中     | 早停选的模型非 test 最优                      | 同时记 test 曲线；试`early_stopping: none` 消融       |

---
## 2.16 与论文基准的「小样本」对比（预算口径对齐）

### 2.16.1 澄清口径：102 是「全量」的数字，可用池只有 ~33

| 口径 | 样本 | 每类（全体）min/中位/均值/max | 每类（treatment） |
|---|---|---|---|
| 全量（所有 split） | 125,510 | 94 / **102** / 110.2 / 492 | 94–102（均值 101.8） |
| train（两 site） | 81,224 | 58 / 66 / 71.3 / 306 | 58–66（均值 65.9） |
| **train site1 = AL 池** | 40,612 | 29 / **33** / 35.7 / 153 | **29–33（均值 33.0）** |
| train site2 = id_test | 40,612 | 29 / 33 / 35.7 / 153 | 29–33 |
| val | 9,854 | 6 / 8 / 8.7 / 32 | 6–8（均值 8.0） |
| test | 34,432 | 24 / 28 / 30.2 / 154 | 24–28（均值 27.9） |
| （控制类） | — | 31 类，全量 408/492 | 池内 4,097 张，128–153/类（均值 132.2） |

> ⚠️ **常见误读**：`102` 是全量数据集的中位数。实验一真正能用的池（`train` site1）
> 里 treatment 类**只有 ~33 张/类**，是全量的 1/3。

### 2.16.2 与论文基准对比

| 数据集/口径 | 池样本 | 类数 | 每类均值 | 相对 CIFAR-100 |
|---|---|---|---|---|
| **RxRx1 AL 池** | 40,612 | **1,139** | **33.0** | **0.066×** |
| RxRx1 池 + id_test | 81,224 | 1,139 | 66.0 | 0.13× |
| RxRx1 全量 | 125,510 | 1,139 | 110.2 | 0.22× |
| CIFAR-10（论文基准） | 50,000 | 10 | 5,000 | 10× |
| CIFAR-100（论文基准） | 50,000 | 100 | 500 | 1× |
| TinyImageNet（论文基准） | 100,000 | 200 | 500 | 1× |

一句话：**类数多 11 倍、每类样本少 15 倍**（wide-and-shallow）。

### 2.16.3 最尖锐的一条：论文的「起点」≈ 我们的「终点」

论文 CIFAR-100 的**初始**标注量 = 2000 / 60 个 InD 类 = **33.3 张/类**；
而 RxRx1 整个池每类**一共只有 33 张**。
即：**RxRx1 上每类的绝对监督量上限，就是论文的起跑线**（论文只用掉每类 6.6% 的池，
我们要用掉 100%）。

### 2.16.4 与论文对齐的三个口径

| 对齐项 | 论文 CIFAR-100 | 当前设计 | 等比例换算 | 建议 |
|---|---|---|---|---|
| 初始标注 / 池 | 4%（2000/50000） | 8.6%（3,480/40,612） | 1,624 | 主实验改成 **~1,600（4%）** |
| 总标注 / 池 | 50%（25,000/50,000） | 28%（11,480/40,612） | **20,306** | 主实验改成 **~20,300（50%）**（如 `q=2000`×9 轮 + 1,600） |
| $t_e$ / 每类池样本 | 10%（50/500） | 15%（5/33） | **3.3** | 保留 5 可接受（偏严），但需在文中说明；或用 3 |

### 2.16.5 三点实质影响（已在协议 §3.5 / §4.3 体现）

1. **训练协议必须偏离论文**：论文 ResNet18 从零训 200 epoch；RxRx1 每类 33 张做不到。
   WILDS 官方本身对 RxRx1 用 ImageNet 预训练 ResNet50 → 主设定 `pretrained: true`，
   「从零训」降级为消融（用于量化预训练的必要性）。
2. **1139 类 head 严重过参数化**：512×1139 ≈ 583k 参数 vs 池 40,612 张 →
   正则化与 **Eq.(4) 类平衡**的作用被放大（正是 JODA 的用武之地）。
3. **`avg_acc` 偏乐观**：$t_e=5$ 时「被发现」的类只有 5 张标注，真实测试精度不会好，
   而论文口径假设「发现即可正确分类」→ 需补严格版副指标。

**正面价值**：RxRx1 比 CIFAR 更贴近论文声称的真实场景（知识不完整 + 标注昂贵），
"1139 类 × 33 张/类 × 批次域偏移"本身就是一个值得单独提出的 **harder OSDAL benchmark**。

### 2.16.6 连带影响：基线可行性

- **Badge** 在 1139 类上可能 OOM（论文在 TinyImageNet 上就已 OOM）；可能需要降维/近似。
- **LLoss** 需要额外训练 loss 模块，在 5 张/类下可能不稳。
- **CoreSet** 特征上 O(n²) 可行，但需要先定特征维度。
- → 基线集合可能需要按可行性裁剪，并在文中说明。

---

## 2.17 口头设计 / 待验证想法（未落地，勿当结论）

> 这一节记录"想做但还没定"的想法，避免丢失。落地前需在 `protocol.md` 中冻结为正式设计。

**(a) 动态 $t_e$（对应假设 H3）** —— 候选规则：

| 规则 | 定义 | 直觉 |
|---|---|---|
| ① 按轮回衰减 | $t_e(t)=\max(t_{\min}, \lceil t_e^{(0)}/(1+t)\rceil)$ | 越往后模型越可信，可降低提升门槛 |
| ② 按标注量比例 | $t_e(t)=\max(t_{\min}, \lceil \alpha \cdot (q/\lvert D_{\text{剩余}}\rvert)\rceil)$ | 保证"每轮平均能推 1 个类" |
| ③ 按类计数分位数 | $t_e(t)=\text{quantile}(\{\text{count}(c)\}_{c\in D}, \rho)$ | 自适应到当前标注分布 |
| ④ 按选择精度反馈 | 若上轮 `selection_precision` 低于阈值 → 提高 $t_e$；反之降低 | 精度低说明 D 掺入 O，需更保守 |
| ⑤ 按"能量/距离密度" | 结合 near-OOD 的能量分布估计"是否已形成簇" | 需要额外统计量，成本较高 |

> 建议先实现 ①②④（廉价、可解释），作为 E1-8 的对照。**注意**：动态 $t_e$ 会改变 `num_classes`
> 增长轨迹，必须与固定 $t_e$ 用**同一评测口径**（同比例、同 seed）比较。

**(b) 严格版 `acc_target`**：只在"已发现类"上计算真实准确率，避免论文口径的乐观假设。
实现上可在离线评测脚本里补（`common/eval_offline.py`）。

**(c) 预算口径对齐**：把初始/总标注比例对齐论文的 4% / 50%（见 §2.16.4），
否则曲线形状与论文不可比。

**(d) `initial_per_class` 扫描**：$\{2, 5, 10\}$。5 张/类对 696 类可能太少，
这一项可能比 InD 比例更影响成败。

**(e) 控制类降采样消融**：把 31 个控制类截到 ~33/类（1,023 张），与 treatment 对齐，
用来分离"数据集自身不平衡"与"JODA 类平衡机制"的贡献。

**(f) 域感知类平衡（为实验三预留）**：Eq.(4) 目前只平衡**类**；RxRx1 的 batch effect 提示
可以把 "class share" 扩展为 "domain/batch share"，做一个 `apply_domain_balance` 并列函数。

**(g) 两阶段标注 / triage 叙事**：论文的 $t_e$ 本质是"标注员甄别新类是否任务相关"的模拟；
RxRx1 上可以考虑显式建模（如先只给"是否新类"，再给细类），但会改变标注成本模型，需谨慎。

---

## 2.18 决策记录

**已定（写入 `protocol.md`）**

| 事项 | 决定 | 依据 |
|---|---|---|
| 场景 | OSDAL（`data_scenario: osal-extending`） | 论文核心场景 |
| 类别方案 | **方案 B**：31 控制类 + treatment 60% → InD 696 / D 443 | 控制类先验已知；EMPTY 不应算"待发现" |
| InD 比例 | 60% 为主，**50% 并列主设定** | 60% 对齐论文；50% 针对 RxRx1 每类仅 33 张 |
| $t_e$ | 固定 **5**（消融 3/10） | 每类池样本的 15%；论文 10% 等比例 ≈3.3 |
| 归一化 | **逐图实例标准化**（I/D/O 一致） | 图像极暗（全局均值 ~0.02–0.06），且要避免 energy 假性可分 |
| 主干 | **ImageNet 预训练** ResNet18 | 33 张/类无法从零训 |
| 训练策略 | `retrain` | AL 公平比较标准 |
| OE | `loss_func: OutlierExposure` + **`triplet_mode: triplet`** | 让 OE 作用于 label<0 的 far-OOD |

**待拍板**

| # | 事项 | 建议 | 记录位置 |
|---|---|---|---|
| B1 | 主比例 60% / 50% / 并列 | 并列 | `protocol.md` §7 |
| B2 | treatment 类挑选：连续 / 随机 | 连续为主 | §7 |
| B3 | 控制类是否降采样 | 否（作消融） | §7 |
| B4 | 划分落地：现算 / json | json | §7 |
| B5 | 总标注量 28% / 50% | 50% | §7 |
| B7 | 是否加严格版 `acc_target` | 是 | §7 |
| B8 | 动态 $t_e$ 具体规则 | 见 §2.17(a) | §7 |

---

## 2.19 环境验证结果（2026-10-04）

自检脚本：`scripts/check_env.py`（环境+数据+GPU）、`scripts/check_data.py`（数据链路）。

### 2.19.1 `joda-cuda` 环境本身正常

| 项 | 结果 |
|---|---|
| Python | 3.10.22（`/data/yuxuan/miniconda3/envs/joda-cuda`） |
| torch / torchvision | **2.7.1+cu128 / 0.22.1+cu128** ✔ 这是 Blackwell(sm_120) 需要的正确构建 |
| 核心依赖 | numpy 1.24.4 / pandas 1.5.3 / PIL 10.4.0 |
| JODA 运行时依赖 | yaml / sklearn / tqdm / mlflow / segmentation_models_pytorch / torchmetrics 齐全 |
| `pyro` | 导入失败（`pyro-ppl==1.8.1` 内部断言 `torch.__version__.startswith("1.")`）。**但全仓 `grep -r "import pyro" JODA/src` 零命中 → 僵尸依赖，不影响实验**，可直接卸载或忽略 |
| RxRx1 数据 | `data/wilds/rxrx1_v1.0` 可访问，125,510 行 metadata，51 个实验目录 |

对比：旧的 `joda` 环境是 torch 1.13.1+cu117，**cu117 没有 sm_120 的 kernel**，在 5090/Blackwell 上永远拿不到 GPU。

### 2.19.2 但宿主机 GPU 目前不可用（**与环境无关**）

```
nvidia-smi       -> GPU0: 0000:16:00.0  Unknown Error   ← 掉卡
cuInit(0)        -> 999  (CUDA_ERROR_UNKNOWN)
cuDeviceGetCount -> 3    (CUDA_ERROR_NOT_INITIALIZED，因 cuInit 已失败)
nvmlInit_v2()    -> 0    (NVML 正常 ← 所以 nvidia-smi 能用)
驱动模块版本      -> nvidia / nvidia_uvm 均为 580.105.08，与用户态一致
容器/占用进程     -> 无（不是 docker，/dev/nvidia* 无进程持有）
GPU1             -> RTX PRO 6000 Blackwell, 97,887 MiB，硬件健康（仅 gnome-shell 占 47 MiB）
```

关键区分：**NVML 能工作但 CUDA 内核态初始化失败**。`CUDA_VISIBLE_DEVICES=1` 或指定 UUID 都救不回来，因为 `cuInit` 在驱动层枚举设备，GPU0 卡死会拖垮整个初始化。

**这不是 `joda`/`joda-cuda` 的差别，任何 conda 环境都跑不了 GPU。** 需要 root 修（按顺序试）：

1. `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` — 重载 UVM，最轻量
2. `sudo nvidia-smi --gpu-reset -i 0` — 重置卡死的 GPU0
3. `sudo reboot` — 掉卡最可靠的解法
4. 修好后 `sudo dmesg | grep -i xid`：若见 `Xid 79 (GPU has fallen off the bus)` 则是硬件/供电问题，需报修

附带：`CUDA_VISIBLE_DEVICES` 曾被设成**空字符串**（历史遗留），会让 torch 报 "CUDA unknown error"，排查时先确认它不是空的。

### 2.19.3 预训练权重落地位置（需要 gitignore）

`torchResnet._resnet` 用 `load_state_dict_from_url(url, model_dir="../Model_Weights")`，该路径**相对运行时 CWD**。从 `JODA/` 启动会在**仓库根**生成 `Model_Weights/resnet18-f37072fd.pth`（45 MB）。已在 `.gitignore` 加入 `Model_Weights/`。

---

## 2.20 JODA 实际接入改动清单（最终版）

共 **5 个已跟踪文件 +44/−5**，外加 1 个新增适配器。其中 3 处是**独立的 JODA bug 修复**（不是注册点）。

| # | 文件 | 改动 | 为什么 |
|---|---|---|---|
| 1 | `data_loaders/classification/rxrx1_data_loader.py` | **新增**：薄适配器 | JODA 要求可调用对象能在 `joda_al...classification` 里 import；实验真实代码全在 `exp/rxrx1/common/`，这里只做 sys.path + 注入 `DatasetScenarioConverter` + 暴露两个变换实例 |
| 2 | `load_classification_dataset.py` 顶部 | +import | 同上 |
| 3 | `dataset_factory` | +`"rxrx1": load_rxrx1` | `dataset_comb_name="rxrx1-ta-ta"` 靠 `startswith("rxrx1")` 命中 |
| 4 | 训练 `trans_map` | +`"rxrx1-ta"` | train loader 会查 `f'{name}-{scenario}'` = `"rxrx1-ta-ta"` |
| 5 | 评测 `trans_map` | +`"rxrx1-ta"` | 池/val/test 会被包上 eval 变换，查的是 `dataset["name"]` = `"rxrx1-ta"` |
| 6 | `dataset_registry.ALL_DATASETS` | +`"rxrx1-ta"` | `parser_io` 有 `assert experiment_config["dataset"] in ALL_DATASETS` |
| 7 | `active_coverage_learning.py` ×2、`LfOSA.py` ×1 | **bug 修复**：`num_classes` 由 `list(models["task"].children())[-1].out_features` 改为 `int(dataset_handler.dataset_config["num_classes"])` | `Resnet18T` 的最后一层是自定义 `ClassificationHead`（MLP），**没有 `out_features`** → `AttributeError`。两者语义相同（模型本来就是用 `dataset_config["num_classes"]` 构造的，且每轮 `update_data_pool` 同步它），改后不再假设头结构 |
| 8 | `Joda_logic.iterate_module` | **bug 修复**：递归调用补上 `mode` 参数 | 见下 |

### 2.20.1 bug #8 的完整因果链（**最关键的一条**）

```python
# 修复前：递归时漏了 mode，默认值 mode="cov" 静默生效
name_list, module_list = iterate_module(child_name, child_module, name_list, module_list)
# 修复后：
name_list, module_list = iterate_module(child_name, child_module, name_list, module_list, mode)
```

而 `is_valid` 里 `mode="cov"` 只认 Conv/Linear，`mode="nac"` 才认 `Sequential`/`AdaptiveAvgPool2d`：

| 模型 | 直系子模块 | 修复前 `get_model_layers(nac)` 键 | 后果 |
|---|---|---|---|
| `custom_models.ResNet18`（JODA 自带） | 含 `layer1..4`(Sequential)、`avgpool` | `Sequential-1..4`、`AdaptiveAvgPool2d-1`、`Linear-1` | 第一层就命中，正常 |
| `torchResnet.ResNet18`（= `Resnet18T`） | 只有 `resnet`/`clsmodel` | 只剩 `Conv2d-1..20`、`Linear-1..4` | **崩** |

崩溃路径：`get_layer_output_nac` 的 `layer_selection` 是**硬编码**的
`["AdaptiveAvgPool2d-1", "Sequential-3", "Sequential-2", "Sequential-1", "Linear-1"]`（L760，且无条件覆盖传入的 `layer_selection`）
→ 与 `Resnet18T` 的键只对得上 `Linear-1`
→ `build_database` 里 `if "Linear" in layer_name and self.use_nac: continue` 把它跳过
→ `SA_batch = []` → `torch.cat([], 1)` → `RuntimeError: expected a non-empty list of Tensors`。

**已验证无回归**：修复后 `custom_models.ResNet18` 的 nac 键名与修复前完全一致
（`['Sequential-1', 'Sequential-2', 'Sequential-3', 'Sequential-4', 'AdaptiveAvgPool2d-1', 'Linear-1']`），
因为它的 `layer1..4`/`avgpool` 本来就在第一层命中，不会进入这条递归；`mode="cov"` 路径下递归本来就该传 `"cov"`，也等同。

> 结论：这两处 bug 是 fork A（`Resnet18T` 预训练分支）与 fork B（`num_classes` 推导 / `iterate_module`）
> 各自引入、但**从未一起运行过**的组合缺陷。任何"用 `Resnet18T` 跑 JODA"的配置都会撞上，与 RxRx1 无关。

### 2.20.2 「必须显式覆盖」的危险默认值

| 键 | 默认值 | 不覆盖的后果 |
|---|---|---|
| `data_scenario` | `"standard"` | 不会走 OSDAL 改造，得不到 D 类/far-OOD |
| `far_dataset` | `"random"` | 会走 stock 噪声分支，往池里塞约 0.611×池大小的噪声 |
| `expanding_th` | `100` | RxRx1 每类约 33 张，永远没有类被提升 |
| `near_classes` | `[6,7,8,9]` | 会被 `apply_open_set_modifications` 覆盖，但显式写 `[]` 更清楚 |
| `far_classes` | `[]` | loader 会校验必须是 `[-1]` |
| `sigmoids` | 全 1 | 见 §2.23(4) |

---

## 2.21 **重要修正**：原 §2.11 的「T2 分支」方案是错的

原计划写的是「加一个 rxrx1 专属分支，**只**把 val/test 包上 eval 变换，池不包」。**这会让能源评分崩溃**：

* `get_unlabeled_pool_loader` 与 `labeled_loader` 都直接吃 `self.train_pool`（Raw），
  JODA 的池**必须**被 eval 变换包住才能产出 tensor；不包就只能拿到 PIL 图。
* 实际设计本来就是对的：池包 eval 变换（提供 `ToTensor`+标准化），已标注子集再叠训练变换。

**真正的约束只有一条**：训练变换必须 tensor-safe 且**不能含 `ToTensor`**（因为它拿到的是已标准化张量）。
90° 旋转与水平翻转都是纯像素置换，先标准化再增强 ≡ 先增强再标准化，因此这个顺序在语义上安全。
已按此实现：`TRAIN_TRANSFORM = Compose([Rotate90(), RandomHorizontalFlip()])`，`EVAL_TRANSFORM = Compose([InstanceStandardize()])`。

顺带用真实代码核实的两条语义（此前只是推测）：

* `ood_stats = (label < 0)*2 + (label >= num_class)`：**D 类算「目标侧」**（值为 1，不是正类），只有 `label == -1` 才是正类。
  所以 labeled 池里没有 far-OOD 时 `optimal_threshold=None`（cycle 0 跳过过滤，与论文一致）。
* `avg_acc = num_classes/all_classes × balanced_acc(InD 测试集)`。`apply_open_set_modifications` 会把
  `dataset_config["num_classes"]` **改写**成 `len(ind_classes)`、而 `all_classes` 保留 I+D —— 这正是缩放因子的来源。
  loader 里 `num_classes` 必须填 **I+D**（I 会被覆盖掉）。

---

## 2.22 冒烟执行记录（2026-10-04，CPU，**EXIT=0**）

* 配置 `configs/exp1_smoke.yml`；方案 `smoke`（I=6, D=4）；far-OOD=F2 噪声 200 张；
  `cycles=2, query_size=200, num_epochs=1, batch_size=8, num_workers=0`
* 日志 `results/logs/smoke_run4.log`（**0 个 ERROR**）；产物 `results/smoke_joda_rxrx1/`
  （`checkpoint.model`/`scenario_state.json`/`scenario_log.json`/`stats_file.csv`）
* 耗时：cycle 0 训练 2.7 s；cycle 1 训练 32 s（412 标注样本）；全程约 95 s

### 逐步断言对照

| 阶段 | 日志/状态观察 | 验证了什么 |
|---|---|---|
| c0 | `Test Acc 0.17677`，`Test Avg Acc 0.10606` = 0.6 × 0.17677 | 缩放因子 $I/(I+D) = 6/10$ 精确成立 |
| c0 | `OOD threshold omitted: labeled pool does not contain both InD and far-OOD samples` | **与论文「cycle 0 跳过过滤」一致**；阈值需要 labeled 池里同时有 InD 与 label<0 |
| c0 | `Selection Precision: 0.04`，`New classes: [7]`，counts 里 `-1: 192` | $t_e=2$ 提升机制生效；无过滤时噪声被大量选中 |
| c1 | `Optimal threshold: -1.7719` | labeled 池有 far-OOD 后，energy+Youden-J 阈值可估 |
| c1 | `Selection Precision: 1.0` | 过滤真正生效，200 个全为真实目标 |
| c1 | `Test Avg Acc 0.09999` = 0.7 × 0.142857 | 提升后缩放因子变为 $I_t/(I+D) = 7/10$，头部扩容链打通 |
| c2 | `New classes: [8, 9, 6]`，`remaining Near classes: []` | D 类**全部被发现** |
| c2 | `Test Avg Acc 0.10303` = 1.0 × 0.10303 | 全部 10 类进入 InD，缩放因子到 1 |
| 终态 | `num_classes 7`、`all_classes 10`、`ind_classes [0,1,2,3,4,5,7]`、labeled 412 = 12+200+200、far-OOD 标注 192 = 200×(1−0.04) | 状态完全自洽，数字逐项对得上 |

### 顺带**实证**了协议 §3.3 对 F2 的判断

纯噪声 far-OOD 在无过滤时会**被优先选中**（precision 0.04，192/200 是噪声），
且噪声本身与真实图为天然可分（阈值一开就到 precision 1.0）。
→ **F2 只适合冒烟；主实验必须用 F1（MNIST→PNG）**，否则过滤这一环被"送分"。
这印证了协议里把 F1 设为主设定、F2 只做链路验证的选择。

---

## 2.23 本轮新踩的坑（通用，后续复用）

1. **`batch_size > L0` + `drop_last=True` → 0 个 batch** → `train_epoch` 里
   `sum(mean_loss)/len(mean_loss)` 触发 `ZeroDivisionError`。
   冒烟里 `L0 = I × initial_per_class = 12`，所以 batch 必须 ≤ 12。主实验 L0=3480/batch 64 无此问题。
2. **崩溃时的次生报错会误导**：`mark_as_failed` 会把实验目录改名成 `-failed`，
   而 TensorBoard 写线程还在跑 → 抛 `FileNotFoundError: .../events.out.tfevents...`。
   **排查时永远先找第一个异常**。
3. `sigmoids` **必须显式给出**：`get_layer_output_nac` 里 `dataset` 只对
   `cifar10`/`cifar100`/`imagenet` 有内置 `sig_alpha`，名字是别的（如 `rxrx1-ta`）会
   `UnboundLocalError`（除非传入的 `sigmoids` 为真值）。
4. **layer 键名 = `类名-序号`**，序号按 `named_children()` 遍历顺序计；`layer_selection` 是硬编码的
   `Sequential-1/2/3 + AdaptiveAvgPool2d-1 (+Linear-1)`。换主干模型时必须先核对这套键名。
5. **dataset 名必须写成 `rxrx1-ta`**（带 `-ta`）。工厂用 `startswith`、变换表用最长前缀，
   两者都依赖这个拼接后的名字。

---

## 2.24 新增待办 / 待调超参

| # | 事项 | 说明 |
|---|---|---|
| N1 | **`sigmoids` 未调优** | 现用 cifar100 那套 `{AdaptiveAvgPool2d-1: 50, Sequential-3: 10, Sequential-2: 1, Sequential-1: 0.005}`。它是 `get_layer_output_nac` 里对层层输出做 `sigmoid(·, sig_alpha)` 的缩放系数，直接影响 SA 特征与 SISOMe 排序。建议在消融里加一档（或先小规模扫一次） |
| N2 | **GPU 未修复前只能 CPU** | 冒烟规模下 CPU 完全够用；主实验 72 run 必须等 §2.19.2 修好 |
| N3 | Stage 1：F1 far-OOD | MNIST→PNG（或其它外部目录），用 `make_splits.py --far-ood dir --far-root <路径> --far-n <数量>` 重新生成 B/A 族划分 |
| N4 | memmap / 图片缓存 | `num_workers=8` 时 PNG 解码是瓶颈（协议 §3.5 已注明） |
| N5 | `SisomQuery.query` 里 `input_size` 硬编码 `(1,3,32,32)` | 非 Joda 路径；若以后用 SISOM 基线需改 |
| N6 | 主实验 matrix + `run_matrix.py` | 见 `protocol.md` §5（6 方法 × 2 方案 × 2 OOD × 3 seeds） |
| N7 | `common/eval_offline.py` | 算严格版 `acc_target`（未发现类按 0 计）、`gen_gap`、`acc_worst_celltype` |

---

## 2.25 当前代码落点（供后续接手）

```
exp/rxrx1/
├── README.md                     伞文档
├── common/
│   ├── splits.py                 类别角色 + WILDS 切分 + 冻结 JSON（不依赖 joda_al）
│   ├── transforms.py             InstanceStandardize / Rotate90 / TRAIN_TRANSFORM / EVAL_TRANSFORM
│   ├── rxrx1_dataset.py          RxRx1ImageListDataset + build_pool_and_splits + load_rxrx1
│   └── __init__.py
└── exp1_osal/
    ├── protocol.md               规范协议（给人看）
    ├── notes.md                  本文件（给 AI / 保全信息）
    ├── configs/
    │   ├── exp1_smoke.yml        冒烟配置
    │   └── splits/               smoke_s0 / B60_s0 / B50_s0 / A60_s0（已冻结）
    ├── scripts/
    │   ├── check_env.py          环境+GPU+数据自检
    │   ├── check_data.py         数据链路自检（20 项断言）
    │   └── make_splits.py        生成/冻结划分
    └── results/                  被 .gitignore 忽略（保留 .gitkeep）

JODA/src/joda_al/data_loaders/classification/rxrx1_data_loader.py   薄适配器（新增）
```

已冻结划分的实际数字（`make_splits.py` 输出，与 `protocol.md` §3.1 一致）：

| 划分 | I | D | 池 | 池内 InD / D | L0 | id_test(本实验类别) |
|---|---|---|---|---|---|---|
| `smoke_s0` | 6 | 4 | 330 | 198 / 132 | 12 | 330 |
| `B60_s0` | 696 | 443 | 40,612 | 26,018 / 14,594 | 3,480 | 40,612 |
| `B50_s0` | 585 | 554 | 40,612 | 22,359 / 18,253 | 2,925 | 40,612 |
| `A60_s0` | 683 | 456 | 40,612 | 22,515 / 18,097 | 3,415 | 40,612 |

（B/A 族目前 `far_ood: null`，等 F1 就绪后重跑 `make_splits.py --far-ood dir ...` 填入。）

---