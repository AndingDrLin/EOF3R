# 技术路线

## Stage 0：项目初始化（第 1 周）

- 目录结构 + 文档 + 环境配置
- 输出：README、docs/、.gitignore、conda 环境

## Stage 1：文献调研（第 2-4 周）

- 6 个方向：3DGS 基础、Object-level 3DGS、Feedforward 3D Recon、场景分割、混合表示、Autonomous Demo
- 输出：lit_review.md 完整索引 + lit_notes/ 下每篇笔记
- 确定每个模块的 baseline 选择

## Stage 2：数据准备与场景分解（第 5-7 周）

- 数据集：ScanNet++（优先）/ Replica / 自拍
- 前景分割：SAM2（自动模式或 box prompt）
- 输出：前景 mask + 背景 mask

## Stage 3：前景 3DGS 重建（第 8-10 周）

- 每个前景物体独立训练 3DGS
- 输入：物体多视图 + mask + 相机参数
- 输出：物体高斯球表示（.ply）

## Stage 4：背景 Feedforward 重建（第 9-11 周）

- VGGT（优先）/ DUSt3R 快速重建背景
- 输入：场景全图 + masked 背景图
- 输出：背景 3D 点云

## Stage 5：前景-背景融合（第 11-13 周）

- 坐标系对齐（same_coords / ICP）
- 渲染融合（alpha blending / depth composite）
- 输出：统一可渲染场景

## Stage 6：Autonomous Demo（第 13-15 周）

- 物体定位 demo（推荐）
- Open3D 或简单 Web 可视化
- 输出：可运行 demo 脚本

## Stage 7：Evaluation & 报告（第 14-16 周）

- PSNR/SSIM（2D 渲染）/ Chamfer Distance（3D）
- Baseline：纯 3DGS、纯 VGGT
- 输出：定量表格 + 可视化对比 + 论文初稿
