# EOF3R

**Efficient Object-level Feedforward 3D Reconstruction with 3D Gaussian Splatting**

本科毕业设计原型系统。

## 核心思路

场景分解 → 前景物体用 3DGS 精细重建 + 背景用 feedforward 模型快速重建 → 融合为统一 3D 场景 → autonomous perception demo。

## 当前状态

- [x] Stage 0：项目初始化 ← 当前
- [ ] Stage 1：文献调研
- [ ] Stage 2：数据准备与场景分解
- [ ] Stage 3：前景 object-level 3DGS
- [ ] Stage 4：背景 feedforward 重建
- [ ] Stage 5：前景-背景融合
- [ ] Stage 6：Autonomous Demo
- [ ] Stage 7：Evaluation & 报告

## 环境

- Python 3.10+
- CUDA 12.x（推荐）
- 详见 `requirements.txt`

## 项目结构

```
EOF3R/
├── docs/           # 项目文档
├── lit_notes/      # 论文阅读笔记
├── experiments/    # 实验记录
├── configs/        # 配置文件
├── scripts/        # 辅助脚本
├── src/            # 核心源码
├── data/           # 数据集（gitignored）
├── outputs/        # 输出（gitignored）
└── tests/          # 测试
```
