# 实验层（`exp/`）

按**课题（数据集）**分目录组织全部实验，与代码库（`JODA/`）、文档（`doc/`）解耦。

```
exp/
└── rxrx1/              # RxRx1 细胞形态数据集上的 JODA 系列实验
    ├── README.md       # ← 从这里开始（总览 + 共用事实 + 代码放置约定）
    ├── common/         # 多实验共用代码（与 JODA 解耦）
    ├── exp1_osal/      # 实验一：OSDAL（类发现 + OOD 过滤）
    ├── exp2_domain_shift/  # 实验二：跨批次域偏移纯 AL
    └── exp3_combined/  # 实验三：域偏移下的 OSDAL（最终形态）
```

## 约定

1. **两层结构**：`exp/<课题>/<实验>/`。每个实验自带：
   - `protocol.md` —— **规范实验协议**（给人看：背景 / 目的与假设 / 设计 / 指标）
   - `notes.md` —— **AI 维护区**（全量探索信息、坑点、实现细节、可复用清单、口头设计）
   - `configs/` + `scripts/` + `results/`
2. **共用代码**放课题层 `common/`，不要在各实验间复制。
3. **只有必须被框架 import 的最小接入代码**才放进 `JODA/src/`；其余全部留在 `exp/`。
4. 新增课题时，照 `rxrx1/` 复制一份目录骨架即可。

> 详细规则见 [`rxrx1/README.md`](rxrx1/README.md)。
