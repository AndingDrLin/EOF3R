# TODO

## Stage 0：项目初始化

- [x] 创建目录结构
- [x] 创建 .gitignore
- [x] 创建 README.md
- [x] 创建 requirements.txt（骨架）
- [x] 创建 docs/project_scope.md
- [x] 创建 docs/roadmap.md
- [x] 创建 docs/lit_review.md
- [x] 创建 docs/standards.md
- [x] 创建 docs/todo.md
- [x] 创建 lit_notes/_template.md
- [x] 创建 experiments/exp_template.md
- [x] 创建 configs/default.yaml
- [x] 创建 src/__init__.py
- [ ] 创建 conda 环境并安装依赖
- [ ] 测试基础环境可运行（import torch, open3d 等）
- [ ] 首次 git commit

## Stage 1：文献调研

- [ ] 阅读 Kerbl et al. "3D Gaussian Splatting" (SIGGRAPH 2023)
- [ ] 阅读 Zhang et al. "Review of Feed-forward 3D Reconstruction" (arXiv 2025)
- [ ] 阅读 Wang et al. "VGGT" (CVPR 2025)
- [ ] 阅读 Zhu et al. "ObjectGS" (ICCV 2025)
- [ ] 阅读 SAM 2 论文 + 官方文档
- [ ] A 方向（3DGS 基础）笔记完成
- [ ] B 方向（Object-level 3DGS）至少 3 篇笔记完成
- [ ] C 方向（Feedforward）至少 3 篇笔记完成
- [ ] D 方向（场景分割）调研完成
- [ ] E 方向（混合表示）至少 3 篇笔记完成
- [ ] F 方向（Autonomous Demo）调研完成
- [ ] 确定各模块 baseline 选择
- [ ] 更新 lit_review.md 为完整状态

## Stage 2：数据准备与场景分解

- [ ] 确定最终数据集
- [ ] 下载数据集
- [ ] 数据预处理脚本
- [ ] SAM2 分割 pipeline 跑通
- [ ] 前景-背景 mask 质量检查

## Stage 3：前景 3DGS 重建

- [ ] 3DGS 官方代码本地跑通
- [ ] 单物体 3DGS 重建流程
- [ ] 多物体独立训练流程

## Stage 4：背景 Feedforward 重建

- [ ] VGGT/DUSt3R 官方代码本地跑通
- [ ] 背景重建流程

## Stage 5：融合

- [ ] 坐标系对齐
- [ ] 渲染融合
- [ ] 融合质量检查

## Stage 6：Demo

- [ ] Demo 脚本
- [ ] 可视化

## Stage 7：Evaluation

- [ ] 定量评估指标计算
- [ ] Baseline 对比
- [ ] 可视化对比图
- [ ] 论文初稿
