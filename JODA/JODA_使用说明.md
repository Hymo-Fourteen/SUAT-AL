# JODA 使用说明 / 排错与改动记录

仓库：https://github.com/TUM-DAML/Joint_Out_of_Distribution_Filtering_and_Data_Discovery_Active_Learning.git

本文件记录在本机（Windows + Miniconda + CPU-only PyTorch）从零把 JODA 真正跑起来一次的过程中
遇到的所有问题，以及对应的决定与代码改动。最终已成功完成一次 `method_type=Joda` 的
Pool 主动学习 smoke run。

---

## 1. 最终可复现的运行方法

### 1.1 克隆仓库

```bash
git clone https://github.com/TUM-DAML/Joint_Out_of_Distribution_Filtering_and_Data_Discovery_Active_Learning.git
cd Joint_Out_of_Distribution_Filtering_and_Data_Discovery_Active_Learning
```

### 1.2 创建 conda 环境

本机 conda 的 TUNA 镜像配置中包含失效的 `pkgs/msys2` channel，直接 `conda create` 会报
`HTTP 404`。本次决定不改动用户全局 `.condarc`，而是在创建环境时临时指定官方 channel：

```bash
conda create -y -n joda python=3.10 --override-channels -c defaults
```

### 1.3 安装依赖

```bash
conda activate joda
python -m pip install -r requirements.txt
```

`requirements.txt` 已被修正并补齐（原文件写错了 PyTorch 包名，且漏了不少运行时依赖）。

### 1.4 运行 JODA

```bash
# Windows Git Bash / PowerShell 均可用；必须让 src 进入 PYTHONPATH
PYTHONPATH=src python src/joda_al/start_active_learning.py --config-path run_configs/joda_smoke_cifar10_test.yml
```

如果使用 `conda run`（无需先 activate）：

```bash
PYTHONPATH=src conda run --no-capture-output -n joda python src/joda_al/start_active_learning.py --config-path run_configs/joda_smoke_cifar10_test.yml
```

说明：

- 配置文件：`run_configs/joda_smoke_cifar10_test.yml`
- 数据集：`cifar10-test`，自动下载到 `./data/cifar10`（本次已下载并校验完成）
- 输出目录：`./experiments/joda_smoke_test`
- 日志：`run_joda.log`
- 为了让 CPU 小任务能在合理时间内完成，smoke 配置使用 `cycles=1`、
  `num_epochs=1`、`batch_size=32`、`query_size=20`、`pretrained=False`。
  完整实验应按论文/文档调大这些参数。

---

## 2. 本次实际运行结果

运行命令最终返回 `EXIT_CODE=0`（最终环境清理后已再次复跑确认）。
最终 `pip check` 输出为 `No broken requirements found.`。

关键输出（节选自 `run_joda.log`）：

```text
Dataset: cifar10-test
Method type:Joda
Cycle type:Pool
cpu
...
Cycles: 0/1
Training Done
Test Acc 0.1953
Test F1 0.15195544071448278
Method used: Joda
Building Coverage..
feature_num:  1482
Finding top coverage samples..
max cov-gain selected: 6.054939270019531
min cov-gain selected: 3.875194787979126
Current Coverage: 0
Training Done
Test Acc 0.204
Test F1 0.11637874089739556
```

产物：

- `experiments/joda_smoke_test/stats_file.csv`：第 0 轮和最终轮指标
- `experiments/joda_smoke_test/stat_statistics_0.csv`：JODA 每样本
  `DISTA/DISTB/DSA/ENERGY/GEN/sample_index/OOD` 统计
- `experiments/joda_smoke_test/checkpoint.model`：最终模型
- `experiments/joda_smoke_test/arg_config.json`、`log.txt`
- `experiments/tf_board/joda_smoke_test/`：TensorBoard 事件

由于 smoke run 只有 1 个 epoch、且未用预训练权重，准确率低（0.1953/0.204）是预期现象；
这仅用于验证整条 JODA 流程可运行，不代表论文性能。

另外补做了一次 CIFAR-100 smoke run（新增 `cifar100-test` 小切分：
1000 labeled + 1000 unlabeled，1 epoch，ResNet18，无预训练，`query_size=50`）：

```text
Dataset: cifar100-test
Method type:Joda
Cycle type:Pool
Test Acc 0.0224   Test F1 0.003674   # 初始轮（CPU）
Test Acc 0.0241   Test F1 0.004521   # 最终轮（CPU）
Test Acc 0.0290   Test F1 0.005590   # 初始轮（CUDA, torch 2.7.1+cu128）
Test Acc 0.0278   Test F1 0.005101   # 最终轮（CUDA, torch 2.7.1+cu128）
```

CIFAR-100 有 100 类，而训练集只有 1000 张、且未用预训练权重，因此约 2%~3% 的
准确率接近随机水平，同样只是 smoke 验证，不是论文配置下的性能。

---

## 3. 遇到的问题与处理

### 3.1 环境与依赖问题

| # | 现象 | 原因 / 决定 | 处理 |
|---|------|-------------|------|
| 1 | `conda create -n joda` 报 `HTTP 404 ... anaconda/pkgs/msys2` | 本机 `.condarc` 里 TUNA 的 `msys2` channel 失效 | 不修改全局 conda 配置，改用 `--override-channels -c defaults` 创建环境 |
| 2 | `pip install -r requirements.txt` 报 `No matching distribution found for pytorch==1.13.1` | 原 `requirements.txt` 写的是 `pytorch`，PyPI 上正确的包名是 `torch` | 将 `pytorch==1.13.1` 改为 `torch==1.13.1` |
| 3 | `import joda_al...` 报 `ModuleNotFoundError: No module named 'mlflow'` | `requirements.txt` 漏了 `mlflow` | 先安装最新 mlflow，结果其把 numpy/scikit-learn/protobuf 升级到与 torch 1.13.1、tensorboard 2.9.1 不兼容的版本；因此改安装并固定 `mlflow==1.30.0`，同时恢复 `numpy==1.24.4`、`scikit-learn==1.0.2`、`protobuf==3.19.6` |
| 4 | 安装 mlflow 后报 `ModuleNotFoundError: No module named 'pkg_resources'` | 新版 conda Python 3.10 自带 setuptools 83，已移除 `pkg_resources`；mlflow 1.30.0 仍需要它 | 固定 `setuptools==65.5.0` |
| 5 | 报 `ModuleNotFoundError: No module named 'torchvision'` | 依赖缺失 | 安装与 torch 1.13.1 匹配的 `torchvision==0.14.1` |
| 6 | 报 `ModuleNotFoundError: No module named 'segmentation_models_pytorch'` | 依赖缺失（语义分割 loss 模块被无条件 import） | 安装 `segmentation-models-pytorch==0.3.3` |
| 7 | `tensorboard 2.9.1 requires protobuf<3.20`，而最新 mlflow 会装 protobuf 6.x | 依赖解析冲突 | 显式固定 `protobuf==3.19.6` |
| 8 | `PyYAML` 虽由 mlflow 间接带入，但不应依赖隐式传递依赖 | 配置解析代码直接 `import yaml` | 在 requirements 中显式加入 `PyYAML==6.0.3` |

`requirements.txt` 最终加入了：`torchvision`、`segmentation-models-pytorch`、`mlflow==1.30.0`、
`setuptools==65.5.0`、`protobuf==3.19.6`、`PyYAML`、`wheel==0.38.4`，并修正包名为 `torch`。

### 3.2 配置解析问题

| # | 文件 | 现象 | 处理 |
|---|------|------|------|
| 9 | `Config/config.py` | 读取 YAML 配置时报 `KeyError: 'args'`；`--method-type` 在 training 和 experiment 两段默认配置中都存在时 argparse 报 conflict | `create_argparser` 对没有 CLI 类型映射的 dataclass 字段（如 `args`、`model_configuration`）直接跳过；同时用 `seen_options` 去重，避免重复注册同一 CLI 参数 |
| 10 | `Config/config_dataclass.py` | `load_experiment_config()` 返回的默认 dict 缺少 `dataset/dataset_path/order/size/static_configuration`，也不含 `method_type/base`，导致 YAML/CLI 中的这些键被 `update_dict(expand=False)` 丢掉，运行时报 `KeyError: 'dataset'` | `DataSetConfig` 是未装饰的 dataclass，`asdict(DataSetConfig())` 取不到类级字段；改为手动合并这 5 个字段；同时在 `ExperimentConfig` 中补上 `method_type`、`base` 两个默认字段 |
| 11 | `Config/parser_io.py` | 使用 `--config-path` 时，`argparse` 中保留的空字符串默认 `experiment_folder=""` 会覆盖 YAML 里的 `joda_smoke_test`，导致结果写到 `./experiments\` 根目录 | 在 `parse_file_cli_hierarchy` 中，若 `arg_config["experiment_folder"]` 为空则将其移除，让 YAML 值生效 |
| 12 | `Selection_Methods/SelectionMethodFactory.py` | `parse_and_validate_config` 报 `AssertionError: No method Joda!`，任何方法都会被误判不存在 | `check_existence()` 第一行 `return keyword in __method__registry__` 导致后面真正扫描逻辑不可达；改为实际调用 `find_modules/find_classes/create_method_map` 再判断 |

### 3.3 CPU / CUDA 硬编码问题

本机有 RTX 5060，但本次通过 PyPI 安装的是 `torch 1.13.1+cpu`，`torch.cuda.is_available()` 为 `False`。
仓库代码大量硬编码 `.cuda()`，并默认选择 `cuda:0`，在 CPU-only torch 上无法运行。
本次决定：保留 CPU 可运行能力，把分类训练路径改成“跟随传入的 `device`”。

| # | 文件 | 处理 |
|---|------|------|
| 13 | `start_active_learning.py` | `CUDA_VISIBLE_DEVICES` 非空时原先无条件生成 `torch.device("cuda:...")`；改为 `torch.cuda.is_available()` 为真才用 CUDA，否则回退 `cpu` |
| 14 | `models/classification/model_utils.py` | `with torch.cuda.device(device)` 改为自定义 `device_context()`（CUDA 时用 `torch.cuda.device`，CPU 时用 `nullcontext()`）；`task_model.cuda()` 改为 `task_model.to(device)`，并给 `classification_model_factory/classification_ensemble_factory` 传入 `device` |
| 15 | `model_operations/pytorch/operations.py` | 训练循环中 `inputs.cuda()` / `labels.cuda()` 和 `torch.cuda.device(device)` 改为 `.to(device)` 与 `device_context()` |
| 16 | `task_supports/pytorch/ClassificationHandler.py` | 评估时 `torch.tensor([]).cuda()`、`inputs.cuda()` 改为 `torch.tensor([], device=device)` 和 `inputs.to(device)` |

如果以后要跑 GPU：应安装与显卡/驱动匹配的 CUDA 版 PyTorch（RTX 50 系列通常需要
CUDA 12.8+ 的较新 torch），并相应重跑依赖安装；上述 device 相关修改在 GPU 下仍兼容。

### 3.4 数据加载问题

| # | 文件 | 现象 | 处理 |
|---|------|------|------|
| 17 | `data_loaders/load_dataset.py` | `load_classification_dataset` 里访问 `dataset["dataset_scenario"]`，但 `load_pool_dataset/load_stream_dataset/load_multi_stream_dataset` 构造的 `dataset` dict 没有该键，报 `KeyError: 'dataset_scenario'` | 三处加载函数都补上 `dataset["dataset_scenario"] = config.experiment_config.get("data_scenario", "standard")` |
| 18 | `data_loaders/load_dataset.py` | `load_task_dataset` 把整个 `Config` 传给 `load_classification_dataset`，后者实际要的是 `experiment_config`，报 `KeyError: 'data_scenario'` | 改为传 `config.experiment_config` |
| 19 | `data_loaders/classification/load_classification_dataset.py` | `cifar10-test` 分支中 `unlabeled_idcs` 覆盖了 100..999，而 labeled 是 0..999，且 training pool 的未标注段实际是 1000..1999；会把已标注样本再“查询”一遍 | 把 `cifar10-test` 的 unlabeled streams 修正为 `[1000, 2000)` 内按 100 个样本切分的 10 段 |
| 20 | CIFAR-10 官方源下载极慢（约 30~50 KB/s，预计 1 小时以上） | 服务器支持 Range 请求 | 编写 `tools/download_cifar10_ranged.py`，用 6 线程分段下载 + 断点续传 + MD5 校验；最终完整下载 `cifar-10-python.tar.gz`（MD5 `c58f...349a` 匹配） |

### 3.5 JODA 选择逻辑本体的问题

| # | 文件 | 现象 | 处理 |
|---|------|------|------|
| 21 | `Selection_Methods/sample_selection.py` | `factory_query` 中 `config.method.args`：`config` 实际是普通 `dict`，没有 `.method` 属性，任何 query 都会 `AttributeError` | 改为安全的 `config.get("method", {}).get("args", {})` |
| 22 | `Selection_Methods/Coverage_Based_Methods/active_coverage_learning.py` | `JodaQuery` 读取 `dataset_handler.dataset_config["size"]`，但 CIFAR 加载器只提供 `sample_size`，报 `KeyError: 'size'` | 优先取 `size`，回退到 `sample_size`（再回退 `(32,32)`） |
| 23 | `Selection_Methods/Coverage_Based_Methods/Joda_logic.py` | `JodaLogic.assess/top_k` 调用 `self.build_greedy(...)`，但类里没有该方法，报 `AttributeError`；这些方法被写在 `SisomLogic` 里 | 在 `JodaLogic` 中补齐 `nac_forward_pass`、`build_greedy`、`calculate_greedy` 三个方法 |
| 24 | 同上 | 新增 `build_greedy` 时 `cached_SA_labels` 默认 float32，写回真实 long 标签时 dtype 不匹配 | 初始化为 `torch.tensor([], dtype=torch.long)` |
| 25 | 同上 | `calculate_greedy` 用同一个变量名 `dist_b_list` 同时表示“每样本到各类别的距离列表”和“所有样本的 dist_b 列表”，导致统计张量列数不一致 | 外层列表重命名为 `all_dist_a/all_dist_b` |
| 26 | 同上 | `JodaLogic.__init__` 缺少 `use_shapley` 属性 | 补上 `self.use_shapley = False` 和 `self.data_count = 0` |
| 27 | 同上 | JODA 内部对缓存样本建的 `DataLoader(..., num_workers=8)` 在 Windows 下不稳定/开销大 | 改为 `num_workers=0` |
| 28 | `defintions.py` | 训练/评估代码引用 `TaskDef.zeroclassification`，但枚举中没有该成员，报 `AttributeError` | 在 `TaskDef` 中补上 `zeroclassification = "zeroclassification"` |

### 3.6 CIFAR-100 附加运行中发现并修复的问题

| # | 文件 | 现象 | 处理 |
|---|------|------|------|
| 29 | `data_loaders/classification/load_classification_dataset.py` | 数据集名匹配顺序中 `cifar10` 排在 `cifar100` 前面，而 Python 的 `"cifar100-test-standard".startswith("cifar10")` 为 `True`，导致所有 `cifar100-*` 数据集都被错误地当作 CIFAR-10 加载 | 把 `cifar100-ub-ta`、`cifar100` 放到 `cifar10` 之前匹配 |
| 30 | `data_loaders/dataset_registry.py` / `classification/load_classification_dataset.py` | 仓库原本没有小规模 CIFAR-100 smoke 切分 | 新增 `cifar100-test` 数据集名和切分逻辑：1000 labeled + 1000 unlabeled，验证集为原 train 后 5000 张 |

新增运行配置与日志：

- `run_configs/joda_cifar100_test.yml`
- `run_joda_cifar100.log`
- `tools/download_cifar100_ranged.py`

### 3.7 CUDA 加速适配与实测

本机有 RTX 5060 Laptop GPU（Blackwell，compute capability `sm_120`），
原来的 `torch 1.13.1` 不支持该显卡，因此另外创建了 `joda-cuda` 环境，
安装 `torch 2.7.1+cu128` / `torchvision 0.22.1+cu128`。

为了兼容 torch 2.x，又做了两个兼容性补丁：

| # | 文件 | 现象 | 处理 |
|---|------|------|------|
| 31 | `Pytorch_Extentions/dataset_extentions.py` | torch 2.7 移除了 `torch.utils.data.dataset.T_co`，导入时报 `ImportError` | 改为 try/except 导入，失败时定义 covariant `TypeVar("T_co")` |
| 32 | `models/model_utils.py`、`models/classification/model_utils.py`、`model_operations/pytorch/operations.py` | torch 2.6+ 的 `torch.load` 默认 `weights_only=True`，加载本仓库含 numpy 对象的 checkpoint 失败 | 三处 `torch.load(...)` 显式加 `weights_only=False`（加载自己生成的 checkpoint，来源可信） |

CUDA 环境文件与实测命令：

```bash
conda env create -f environment_cuda.yml
conda activate joda-cuda
python -m pip install -r requirements-cuda.txt

PYTHONPATH=src python src/joda_al/start_active_learning.py --config-path run_configs/joda_smoke_cifar10_test.yml
```

实测效果（同一 CIFAR-10 smoke 配置）：

```text
CPU  torch 1.13.1：单 epoch 约 51 s，完整 smoke 约 4 分半
GPU  torch 2.7.1+cu128：单 epoch 约 5 s，完整 smoke 约 30 s
设备日志：cuda:0
```

CUDA 版 CIFAR-10 smoke 结果为 `Test Acc 0.1799 -> 0.1862`；
CIFAR-100 为 `Test Acc 0.0290 -> 0.0278`。
注意：改用 torch 2.x 后，随机初始化/浮点 kernel 与 torch 1.13 不同，
因此相同配置下的指标会有小幅变化，这是正常的。

### 3.8 完整 CIFAR-100 训练的尝试（尚未达到论文级准确率）

在 `joda-cuda` 环境下额外尝试了标准完整 CIFAR-100（train 50k / test 10k，
`base=Full`，batch 256，SGD，CUDA）：

- 仓库自带的 `custom_models.ResNet18`（从随机初始化开始，约 35 个 epoch）：
  `Test Acc` 最高约 **0.581**。
- 新增 `Resnet18T` 选项（torchvision 风格 ImageNet 预训练，10 个 epoch）：
  `Test Acc` 约 **0.549**。

期间又发现并修复/记录了：

| # | 文件 | 现象 | 处理 |
|---|------|------|------|
| 33 | `models/classification/model_utils.py` | `pretrained=True` 实际上被忽略：`"Resnet18"` 走的是 `custom_models.ResNet18`，该函数没有 pretrained 逻辑 | 新增 `"Resnet18T"` 分支，使用 `torchResnet.ResNet18` 并真正加载 ImageNet 权重 |
| 34 | `data_loaders/classification/load_classification_dataset.py` | 标准 `cifar100`/`cifar100-standard` 没有任何训练数据增强 | 补上 `RandomHorizontalFlip + RandomCrop(32, padding=4)` |

**为什么仍不接近论文**：

1. 论文 JODA 配置不是普通 100 类分类，而是 `cifar100-ta` + open-set AL +
   `places365F` far-OOD 数据集，且 `num_epochs=200`、多轮 AL cycle。
2. 本机是 8GB 笔记本 GPU，而每个工具调用还有 10 分钟上限，无法在一次会话内完成
   200 epoch × 多 cycle 的训练。
3. 原仓库训练/增强管线本身还有缺陷（如 `pretrained` 被忽略、增强后处理顺序问题），
   不修完整训练协议就难以复现论文精度。

因此目前只能说“完整数据规模的普通训练能跑、当前约 58%”，不能说“接近论文”。

### 3.9 其他说明（非致命）

- `start_active_learning.py` 启动时会打印
  `Warning: Could not import 'patches' for 3DDet.`，这是缺少可选 KECOR 3D 检测补丁，
  分类任务不受影响。
- 训练初期 sklearn 会打印 `Precision is ill-defined...`：1 个 epoch 的小模型在验证集上
  某些类没有预测样本，属正常 warning，不影响运行。
- 失败重试时，`start_experiments` 的异常处理会尝试把 experiment 目录改名为 `-failed`；
  在 Windows 上若 TensorBoard/文件句柄尚未释放，会叠加一个 `PermissionError`。
  本次通过每次重跑前手动清理 `experiments/joda_smoke_test*` 和
  `experiments/tf_board/joda_smoke_test*` 规避。更稳妥的长期修复是让该 rename 失败时
  不掩盖原始异常。

---

## 4. 本次修改的文件清单

代码修改（`git status` 中的 `M`）：

```text
requirements.txt
src/joda_al/Config/config.py
src/joda_al/Config/config_dataclass.py
src/joda_al/Config/parser_io.py
src/joda_al/Selection_Methods/Coverage_Based_Methods/Joda_logic.py
src/joda_al/Selection_Methods/Coverage_Based_Methods/active_coverage_learning.py
src/joda_al/Selection_Methods/SelectionMethodFactory.py
src/joda_al/Selection_Methods/sample_selection.py
src/joda_al/data_loaders/classification/load_classification_dataset.py
src/joda_al/data_loaders/dataset_registry.py
src/joda_al/data_loaders/load_dataset.py
src/joda_al/defintions.py
src/joda_al/model_operations/pytorch/operations.py
src/joda_al/models/classification/model_utils.py
src/joda_al/models/model_utils.py
src/joda_al/Pytorch_Extentions/dataset_extentions.py
src/joda_al/start_active_learning.py
src/joda_al/task_supports/pytorch/ClassificationHandler.py
```

新增文件：

```text
JODA_使用说明.md                                  # 本文档
environment.yml                                   # CPU 环境定义
environment_cuda.yml                              # CUDA 环境定义
requirements-cuda.txt                             # CUDA 12.8 依赖集合
run_configs/joda_smoke_cifar10_test.yml           # CPU/CUDA smoke-run 配置（CIFAR-10）
run_configs/joda_cifar100_test.yml                # CPU/CUDA smoke-run 配置（CIFAR-100）
run_configs/joda_cifar100_full_pretrained.yml     # 完整 CIFAR-100 Full 训练配置（分阶段）
run_configs/joda_cifar100_full_sgd.yml            # 完整 CIFAR-100 SGD 续训配置
run_configs/joda_cifar100_full_pretrained_t.yml   # Resnet18T ImageNet 预训练 Full 配置
tools/download_cifar10_ranged.py                  # 分段续传下载 CIFAR-10 工具
tools/download_cifar100_ranged.py                 # 分段续传下载 CIFAR-100 工具
run_joda.log                                      # CIFAR-10 CPU 成功运行日志
run_joda_cifar100.log                             # CIFAR-100 CPU 成功运行日志
run_joda_cuda.log                                 # CIFAR-10 CUDA 成功运行日志
run_joda_cifar100_cuda.log                        # CIFAR-100 CUDA 成功运行日志
run_joda_cifar100_full.log                          # 完整 CIFAR-100 训练日志
run_joda_cifar100_pretrained_t.log                  # Resnet18T 预训练日志
data/                                             # CIFAR-10 数据
experiments/                                      # 运行产物
```

---

## 5. 跨平台 / Linux 复现说明

不要把本机 `E:\A\Anaconda\miniconda3\envs\joda` 直接复制给 Linux 用户：
conda 环境里包含 Windows 专用包（如 `vc`、`vs2015_runtime`、`pywin32`），
且绝对路径、可执行文件布局都和 Linux 不兼容。

正确做法是“同步代码 + 重建环境”。本次修改后的源码和配置基本是平台无关的，
并且新增了 `environment.yml`。

Linux 用户建议按以下顺序操作：

```bash
# 1. 同步整个仓库目录；如果 data/ 没同步，需要让脚本重新下载 CIFAR-10
git clone <仓库地址/或使用同步后的目录>
cd Joint_Out_of_Distribution_Filtering_and_Data_Discovery_Active_Learning

# 2. 重建环境
conda env create -f environment.yml
conda activate joda
python -m pip install -r requirements.txt
python -m pip check

# 3. 从仓库根目录运行
PYTHONPATH=src python src/joda_al/start_active_learning.py --config-path run_configs/joda_smoke_cifar10_test.yml
```

Linux 下的注意事项：

- 本次只在 Windows CPU-only torch 上验证过；Linux 全流程尚未在干净机器上实测，
  因此应把上面的步骤当作“高概率可运行的手册”，而不是“已经做过 CI 验证的保证”。
- `requirements.txt` 中所有直接依赖都有 Python 3.10 的 Linux wheel，
  理论上 `pip install` 可完成；但间接依赖（尤其 `timm` 拉取的
  `huggingface-hub`）没有锁死，未来仍可能出现新的依赖漂移。
- Linux 使用 PyPI 的 `torch==1.13.1` 通常是 CUDA 11.7 版本；若没有 NVIDIA 驱动，
  我们修改后的代码会自动回退到 CPU 并运行。若需要纯 CPU wheel，可改用
  `pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`。
- 如果 Linux 机器是较新 GPU（如 RTX 40/50 系列），`torch 1.13.1` 可能太老，
  需要换用支持该显卡 CUDA 版本的新 torch/torchvision；但 JODA 其他依赖和代码
  未针对新 torch 完整验证。
- 本文档修改的只是 `method_type=Joda` 的 classification/Pool smoke 路径；
  语义分割、3D 检测以及其他 query 方法仍可能保留原始仓库中的 CUDA/路径硬编码问题。

---

## 6. 结论

JODA 仓库原始代码在当前 Windows + Python 3.10 + CPU-only PyTorch 环境下无法直接运行，
问题集中在：依赖清单不完整、配置解析丢键、分类路径硬编码 CUDA、以及 JODA 覆盖率逻辑
缺失关键方法。按本文第 3 节的修改后，仓库可以完整执行：

`Joda 主动学习查询 -> 样本选择 -> 标注集更新 -> 最终模型重训与测试` 的流程，
并产出 CSV/TensorBoard 日志与 checkpoint。
