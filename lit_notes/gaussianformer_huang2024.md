# GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction

- **作者**：Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, Jiwen Lu
- **会议/期刊**：ECCV 2024 (Lecture Notes in Computer Science, Vol. 15085)
- **年份**：2024
- **链接**：[ECCV Poster](https://eccv2024.ecva.net/virtual/2024/poster/2107)
- **代码**：[github.com/huang-yh/GaussianFormer](https://github.com/huang-yh/GaussianFormer)
- **阅读日期**：2025-06-16

## 一句话总结

首次用稀疏 3D 语义 Gaussians（非 voxel grid）做自动驾驶的 3D 占据预测：每个 Gaussian 代表一个灵活的关注区域并携带语义特征，通过 attention 从图像聚合信息，再通过 CUDA 加速的 Gaussian-to-voxel splatting 生成密集占据输出，显存消耗仅为现有方法的 17.8%-24.8%。

## 核心方法

1. **Sparse 3D Semantic Gaussians**：用少量 Gaussian（而非密集 voxel）表示场景，每个 Gaussian 有 3D 位置、协方差、语义特征向量。Object-centric 表示天然避免在空白区域浪费计算。
2. **Image-to-Gaussian Attention**：跨注意力从 2D 图像特征聚合信息到 3D Gaussian，迭代细化 Gaussian 属性。
3. **Gaussian-to-Voxel Splatting**：高效的 CUDA 加速 splatting 将稀疏 Gaussian 表示转换为密集占据 voxel grid（供下游使用）。

## 关键数字

| 指标 | 值 |
|------|-----|
| 显存 vs 现有方法 | 17.8%-24.8%（减少 75-82%） |
| 数据集 | nuScenes, KITTI-360 |
| 语义占据精度 | 与 SOTA 可比 |

## 与本文的关系

**高度相关，是本项目"Gaussian → 占据"路线的直接前驱**。GaussianFormer 证明了用 Gaussian 做占据预测（而非渲染）是可行的，其 Gaussian-to-Voxel Splatting 模块与本项目的 BEV 投影在概念上完全一致。核心区别：GaussianFormer 做的是 camera-only BEV 占据（自动驾驶场景），本项目做的是 object-level Gaussian occupancy → BEV footprint（机器人导航场景）。

## 可用性

- [x] 代码开源
- [ ] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- GaussianFormer 是"将 Gaussian 用于占据而非渲染"这一思想的先行者，为本项目的可行性提供了重要背书。
- 后继工作 **GaussianFormer-2** (2025) 进一步引入概率化 Gaussian 叠加，提升了占据预测效率。可以作为参考。
- 项目的 Gaussian-to-Voxel Splatting 的 CUDA 实现可能可以被直接复用或参考，加速我们的 BEV 投影实现。
- GaussianFormer 的 object-centric 设计与我们的 per-object Gaussian 组有天然的契合——每个物体的 Gaussian 组直接对应一个"关注区域"。
