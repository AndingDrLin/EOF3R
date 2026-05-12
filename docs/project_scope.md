# 项目范围与目标

## 标题

Efficient Object-level Feedforward 3D Reconstruction with 3D Gaussian Splatting

本科毕业设计原型系统。

## 要解决的问题

现有 3D 重建 pipeline 对全场景用统一策略，存在矛盾：
- 3DGS 精细但浪费高斯球在背景上，泛化弱；
- Feedforward 模型（VGGT/DUSt3R）快且泛化强，但细节不如 3DGS。

**核心 idea**：按语义角色分前景和背景，各用最优策略重建，再融合。

## 做的（scope 内）

1. object-aware 混合 3D 重建 pipeline（前景 3DGS + 背景 feedforward）
2. 前景-背景融合策略（不同表示的坐标对齐与 rendering 融合）
3. autonomous perception prototypical demo（物体定位或场景理解）
4. 与纯 3DGS / 纯 feedforward 的定量对比

## 不做的（非 scope）

- 不做完整自动驾驶/机器人系统
- 不做实时在线 SLAM
- 不做 dynamic scene / 4D 重建
- 不做 open-vocabulary 检测
- 不发明新 3DGS 变体、新 feedforward 架构

## 研究 vs 工程

| 研究 | 工程集成 |
|------|----------|
| 前景-背景融合策略 | 调用 SAM2 做分割 |
| 不同表示对齐方法 | 调用 3DGS 官方实现 |
| 混合 pipeline 整体方案 | 调用 VGGT/DUSt3R 官方模型 |
| | demo 搭建与评估 |

## 成功标准

1. 至少 2-3 个场景上完成完整 pipeline
2. 融合场景无明显错位
3. 与 baselines 有定量对比（PSNR / Chamfer / time）
4. Demo 可运行展示
