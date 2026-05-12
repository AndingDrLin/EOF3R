# 项目规范（补充）

> **权威来源**：`CLAUDE.md` 是项目宪法，所有规范以它为准。
> 本文仅补充 CLAUDE.md 中未展开的细节，或作为快速查阅。

## 文件命名细节

- 论文笔记：`{模型名}_{作者}{年份}.md`，如 `vggt_wang2025.md`
- 实验记录：`{YYYY-MM-DD}_{简短描述}.md`，如 `2025-06-15_fg_3dgs.md`
- Checkpoint：`{模型名}_{数据集}_{epoch}.pth`，如 `3dgs_scannet_chair_7000.pth`

## 实验命名细节

格式：`{stage}_{日期}_{简短描述}_{关键参数}`

示例：`s3_20250701_obj3dgs_chair_lr0.01`

必记录：配置路径、数据集、随机种子、关键超参、结果数字、失败原因。

## 论文笔记细节

使用 `lit_notes/_template.md`。每篇不超过 1 页。必须包含「与本文的关系」字段，记录复现状态。

## 结果保存细节

```
outputs/results/{exp_name}/
├── config.yaml       # 配置副本
├── metrics.json      # 定量结果
├── visualizations/   # 可视化
└── checkpoints/      # 权重
```

## 可复现性细节

- 固定 random seed（默认 42）
- 每个实验保存完整配置文件
- 记录 GPU 型号和 CUDA 版本
- 预处理脚本纳入版本控制
- 有随机性时跑 3 次取平均
