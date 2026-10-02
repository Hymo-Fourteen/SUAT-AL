# JODA 代码合并说明（A 版 + B 版 → 统一版本）

本文档记录把两份各自基于上游 JODA 代码做的本地修改合并成**一份统一代码**的过程、
冲突的取舍依据，以及验证结果。

## 0. 背景与三方基线

上游仓库（共同基线）：

```
https://github.com/TUM-DAML/Joint_Out_of_Distribution_Filtering_and_Data_Discovery_Active_Learning
commit c0905b9  "Initial Cleaned Up version of ... Code Release"
```

工作区内两份修改：

| 代号 | 目录 | 说明 |
|------|------|------|
| **A** | `JODA/`（本目录，被 SUAT-AL 仓库跟踪） | 一方对代码的修改 |
| **B** | `JODA-2026-9-30/Joint_Out_of_Distribution_Filtering_and_Data_Discovery_Active_Learning/` | 另一方对代码的修改（内含一个只带上游初始提交的 git 仓库，改动尚未提交） |
| **base** | 上游初始提交 | 三方合并的公共基线（用 `git archive HEAD` 从 B 的 git 仓库导出） |

合并方式：以 base 为基线，A、B 各作一条分支，做标准 git 三方合并，
**冲突逐个人工判断、尽量同时保留双方意图**，而不是简单二选一。

---

## 1. 双方改动概要

### 1.1 A 版改动（已记录于 `JODA_使用说明.md`）

主题：**把上游代码在当前环境下真正跑起来**。

- 依赖：修正 `requirements.txt`（`pytorch`→`torch`，补齐 torchvision / mlflow /
  segmentation-models-pytorch / protobuf / PyYAML 等）。
- 配置解析：`create_argparser` 跳过无 CLI 类型映射的字段并去重；
  `load_experiment_config` 补回 `DataSetConfig` 字段；`--config-path` 时避免空的
  `experiment_folder` 覆盖 YAML 值；修复 `check_existence` 使方法注册表可用。
- CPU/CUDA 解耦：`start_active_learning.py`、`models/classification/model_utils.py`、
  `model_operations/pytorch/operations.py`、`ClassificationHandler.py` 中去掉硬编码
  `.cuda()`，改为跟随传入 `device`。
- 数据加载：补 `dataset["dataset_scenario"]`；`load_classification_dataset` 传入
  `experiment_config`；修正 `cifar10-test` 切分；`cifar100` 匹配顺序修正；
  新增 `cifar100-test` 小切分。
- JODA 逻辑：`sample_selection` 取参修正；`active_coverage_learning` 读取 `size`
  回退 `sample_size`；`Joda_logic` 补齐 `nac_forward_pass`/`build_greedy`/
  `calculate_greedy` 等方法；`defintions.py` 补 `zeroclassification`。
- torch 2.x 兼容：`T_co` try/except 导入；`torch.load(..., weights_only=False)`；
  新增 `Resnet18T` 预训练分支。
- 新增若干运行配置、下载工具、使用说明文档。

### 1.2 B 版改动

主题：**扩展数据集/方法 + 修正若干算法与工程细节**。

- 新增数据集：`SYKE-ZooScan` / `SYKE-IFCB` 浮游生物数据集
  （`plankton_dataloader.py`、`dataset_registry.py`、`load_classification_dataset.py`，
  以及 `data_config` 配置字段）。
- 新增调试管线：`utils/pipeline_debug.py`，并接入训练/评估的
  `debug_pipeline` / `debug_epochs` / `debug_max_modules` 配置。
- CIFAR 训练增强：`NormalizedCIFARTrainTransform`（在已归一化张量上做
  crop/flip，padding 用 `-mean/std`），并用 `_lookup_dataset_transform` 支持
  `cifar100-ta-ta` 这类名字。
- JODA 算法修正：`active_coverage_learning` 中 Eq.(4) 的 `final_std` 改为按
  SISOMe 分数的标准差计算；OOD 过滤为空/不足时的健壮性处理；
  `Joda_logic` 中新增 `get_separation_metric` / `get_ind_candidate_mask` /
  `apply_class_balance` 与阈值缺失处理。
- `Pytorch_Extentions/dataset_extentions.py`：为包装器补 `__getitems__`（torch 2.x
  批量取样路径），并自行定义 `TypeVar("T_co")`。
- 其它：`nesterov` / `detect_anomaly` 配置与实现、`query_scenario` 配置、
  `--config_path` 别名、`method_type` 回退、`Pool.py`、`diversity_based`、
  `kcenterGreedy`、`LfOSA`、`sampling_based`、`custom_models`、`loss_functions` 等。
- 新增 `requirements_befor.txt`（改前备份）。

---

## 2. 冲突解决记录（8 个文件）

| 文件 | 冲突点 | 处理方式 |
|------|--------|----------|
| `requirements.txt` | 双方都改依赖 | 取 **A 版**（A 是 B 改动（删除错误的 `pytorch==1.13.1`）的超集，另补齐 torch/torchvision 等） |
| `Config/parser_io.py` | 都处理空 `experiment_folder` 覆盖问题 | 合并条件：`if "--experiment-folder" not in sys.argv or not arg_config.get("experiment_folder")`，同时保留 B 的 `--config_path` 别名、`method_type` 回退、`query_scenario` 打印 |
| `Pytorch_Extentions/dataset_extentions.py` | `T_co` 导入方式 | 取 A 的 try/except 导入（兼容新旧 torch），并保留 B 新增的 `__getitems__` |
| `Selection_Methods/sample_selection.py` | 取方法参数的方式不同 | 以 A 的 `config["method"]["args"]` 为主，追加 B 的扁平 `config["args"]` 回退，两者都兼容 |
| `.../active_coverage_learning.py` | `input_size` 取 `size` vs `sample_size` | 取 A 的「`size` 优先、回退 `sample_size`、再回退 `(32,32)`」，并保留 B 的 Eq.(4) 与 OOD 健壮性改动 |
| `data_loaders/load_dataset.py` | ① `load_task_dataset` 中双方各自转换 config ② `dataset_scenario` 取值 | ① 改为局部变量 `experiment_config`，分类传 `experiment_config`、分割仍传完整 `config`（消除自动合并产生的重复取值 bug）② 保留 A 的 `static_configuration` 行 + B 的 `dataset_scenario → static_configuration → data_scenario → "standard"` 回退链 |
| `.../load_classification_dataset.py` | ① cifar100/cifar10 匹配顺序注释 ② `cifar*` 训练增强 | ① 保留 A 的说明注释 ② 保留 B 的 `NormalizedCIFARTrainTransform` 机制，同时保留 A 新增的 `cifar100` / `cifar100-standard` 增强项（B 的机制未覆盖这两个名字） |
| `task_supports/pytorch/ClassificationHandler.py` | 评估循环里 device 处理 vs 调试钩子 | 合并：A 的 `cuda_ctx` + `.to(device)`，配合 B 的 `enumerate(tqdm(...))` 与 debug 钩子 |

### 2.1 自动合并中的语义冲突（已额外修正）

- **`Config/config_dataclass.py`**：A 因 `DataSetConfig` 当时不是 dataclass 而手写合并 5 个字段，
  B 则给 `DataSetConfig` 加了 `@dataclass` 并新增 `data_config` 字段。直接自动合并会**丢掉
  `data_config`**，而 `plankton_dataloader.py` 依赖它。已改回
  `experiment_config.update(asdict(DataSetConfig()))`（在 B 的 `@dataclass` 下可正常展开全部字段）。
- **`data_loaders/load_dataset.py`**：双方都改了同一函数的 config 取值，自动合并产生了
  “先 `config = config.experiment_config` 再 `config.experiment_config`” 的重复访问 bug，已按上表修正。

---

## 3. 验证结果

- `python -m compileall JODA/src` 通过，无冲突标记残留。
- Pylance 对 `JODA/src` 全量检查：无错误。
- 端到端 CPU 冒烟测试（`joda` 环境，torch 1.13.1，`CUDA_VISIBLE_DEVICES=""`）：

  ```
  PYTHONPATH=src python src/joda_al/start_active_learning.py \
      --config-path run_configs/joda_smoke_cifar10_test.yml
  ```

  完整跑通「Joda 查询 → 样本选择 → 标注集更新 → 重训与测试」，日志可见
  `Building Coverage..` / `Finding top coverage samples..` /
  `Joda candidates after OOD filtering: 1000/1000`（B 的 OOD 过滤日志）/
  `Test Acc ...`，流程正常结束。

---

## 4. 后续建议

- `JODA-2026-9-30/` 含一个内嵌 `.git` 且与 `JODA/` 重复，建议合并确认后删除或加入 `.gitignore`。
- B 的独立运行配置已并入 `JODA/config_standalone/`（其中 `dataset_path` 等仍是 B 机器的绝对路径，
  使用前需按本机修改）。
- 语义有分歧的两处（`dataset_scenario` 取值、`cifar100` 训练增强）已按「尽量兼容双方」处理；
  若你更倾向某一方的原始语义，可在本文件基础上微调。
