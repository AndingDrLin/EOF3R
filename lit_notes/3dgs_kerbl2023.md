# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

- **作者**：Bernhard Kerbl*, Georgios Kopanas*, Thomas Leimkühler, George Drettakis (*equal contribution)
- **会议/期刊**：SIGGRAPH 2023 (ACM Transactions on Graphics, Vol. 42, No. 4)
- **年份**：2023
- **链接**：[Project Page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [Paper](https://dl.acm.org/doi/abs/10.1145/3592433)
- **代码**：[github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **阅读日期**：2025-06-16

## 一句话总结

用显式的 3D 高斯球替代 NeRF 的 MLP 网络表示场景，配合可微 tile-based 光栅化器，实现实时（100+ FPS）高质量新视角渲染，训练时间从 NeRF 的数十小时缩短到 30-40 分钟。

## 核心方法

1. **3D Gaussian 表示**：每个 Gaussian 由位置、协方差矩阵（分解为缩放+四元数旋转）、透明度 α、球谐系数（view-dependent color）定义。显式、可微，结合了体积表示的连续性和点表示的效率。
2. **自适应密度控制**：从 SfM 稀疏点云初始化，迭代优化过程中对重建不足区域 clone/split Gaussian，对过度重建区域剪枝，每 3000 次迭代重置透明度以消除"漂浮物"。
3. **Tile-based 快速光栅化器**：将屏幕分 16x16 tile，GPU radix sort 按深度排序 Gaussian，逐 tile α 混合，到达饱和即停止。反向传播完全可微。

## 关键数字

| 指标 | 值 |
|------|-----|
| PSNR (Mip-NeRF360) | 27.21 |
| SSIM (Mip-NeRF360) | 0.815 |
| LPIPS (Mip-NeRF360) | 0.214 |
| FPS (1080p) | 100+ |
| 训练时间 | ~30-40 min |
| 典型显存 | ~734 MB/场景 |

## 与本文的关系

本项目以 3DGS 为基础表示形式，重点利用其**几何信息**（Gaussian center, scale, rotation, opacity）做占据估计和 BEV 投影，**不追求 photorealistic rendering**（球谐系数和 view-dependent color 仅作诊断输出，不作为核心目标）。3DGS 的 explicit primitive 天然适合提取物体 footprint 和占据形状——这是本项目选择 3DGS 而非 NeRF 的关键原因。

## 可用性

- [x] 代码开源
- [x] 权重可下载（7K/30K iterations）
- [ ] 已在本地跑通
- [x] 显存要求可接受（<12GB）

## 笔记

- 与 NeRF 最大的架构差异：3DGS 是 explicit representation，每个 primitive 有明确的 3D 位置、大小、朝向——这对规划（需要具体几何尺寸）至关重要。
- Gaussian 的 scale/rotation 可以直接转换为 3D bounding box，是 BEV footprint 提取的基础。
- 原始 3DGS 需要逐场景优化（7K-30K 迭代），这对机器人场景不适用。后续的 MVSplat/pixelSplat 解决了 feedforward 问题。
- 球谐系数占存储的大头（每个 Gaussian ~59 维），本项目如果只保留几何字段可以大幅压缩。
