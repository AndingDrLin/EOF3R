# VGGT: Visual Geometry Grounded Transformer

- **作者**：Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny
- **会议/期刊**：CVPR 2025 (Best Paper Award)
- **年份**：2025
- **链接**：[arXiv:2503.11651](https://arxiv.org/abs/2503.11651) | [Project Page](https://vgg-t.github.io/)
- **代码**：[github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- **阅读日期**：2025-06-16

## 一句话总结

一个纯前馈 Transformer（~1.2B 参数），从 1 到数百张图像中一次性预测场景的所有关键 3D 属性（相机内外参、深度图、点云、3D 点跟踪），无需 Bundle Adjustment 或任何几何优化，速度比 DUSt3R 快 10-100 倍。

## 核心方法

1. **Alternating Attention 机制**：24 层 Transformer，交替执行 frame-wise self-attention（建模单图内结构）和 global self-attention（跨图几何关系融合），不需要 cross-attention 层。
2. **多任务预测头**：Camera Head（预测外参四元数+平移+内参 FoV）、Dense Prediction Head（DPT decoder 预测深度图+点云+置信度）、Tracking Head（CoTracker2-based 跨帧点跟踪）。
3. **极度简化**：无 3D-specific inductive bias，所有几何先验从数据中学习。DINO 编码器做 patch embedding。
4. **训练**：64x A100 训练 9 天（160K steps），20+ 数据集混合。

## 关键数字

| 指标 | 值 |
|------|-----|
| 参数 | ~1.2B |
| 4 帧推理速度 | 0.07s (H100) |
| 10 帧推理速度 | 0.14s |
| 4 帧显存 | 2.45 GB (H100) |
| 相机位姿 AUC@30° (Co3Dv2) | ~90% |
| 速度 vs DUSt3R/MASt3R | 10-100x faster |

## 与本文的关系

**本项目背景模块的首选方案**。VGGT 的前馈特性完美匹配我们的"看几秒就要用"的机器人场景——不需要任何逐场景优化。输入 2-4 张场景图，直接输出 dense pointmap + 相机位姿 + 深度图，这构成了背景几何估计的全部输出。VGGT 预测的 pointmap 还可以作为前景 feedforward decoder 的 geometry hint。

## 可用性

- [x] 代码开源
- [x] 权重可下载（含商业可用版本 VGGT-1B-Commercial）
- [ ] 已在本地跑通
- [ ] 显存要求可接受（~2.5GB 4帧，但大帧数需要更多）

## 笔记

- CVPR 2025 Best Paper Award，是当前 feedforward 3D 重建方向的制高点。
- 关键优势：**纯前馈 + 端到端统一**，一个模型取代了传统 SfM（特征提取+匹配+BA+MVS）整个 pipeline。
- 对本项目的独特价值：VGGT 提供的不只是 pointmap，还有相机位姿和深度——这正好是前景 MVSplat（需要位姿）和坐标对齐都需要的关键信息。
- 显存方面：4 帧仅需 2.45GB，完全适合 12GB VRAM 目标。
- 备选方案：如果 VGGT 不便使用（1.2B 参数下载/部署较复杂），DUSt3R 和 MASt3R 是可靠的 fallback。
