# Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields

- **作者**：Shijie Zhou*, Haoran Chang*, Sicheng Jiang*, Zhiwen Fan, Zehao Zhu, Dejia Xu, Zhangyang Wang, Pradyumna Chari, Suya You, Achuta Kadambi (* equal contribution; UCLA / UT Austin / DEVCOM Army Research Lab)
- **会议/期刊**：CVPR 2024
- **年份**：2024
- **链接**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.html)
- **代码**：[github.com/ShijieZhou-UCLA/feature-3dgs](https://github.com/ShijieZhou-UCLA/feature-3dgs)
- **阅读日期**：2025-06-16

## 一句话总结

首个将 2D 基础模型特征（LSeg、SAM）蒸馏到 3DGS 的工作：通过并行 N 维 Gaussian 光栅化器同时渲染 RGB 和 N 维 feature map，用 1x1 卷积 speed-up module 将低维渲染特征上采样到教师特征维度，实现 2.7x 加速训练和 23% mIoU 提升。

## 核心方法

1. **Parallel N-dimensional Gaussian Rasterizer**：扩展 3DGS 的光栅化器，使其可以并行渲染 RGB (3 通道) 和任意 N 维特征 channel。核心实现是避免 warp divergence。
2. **Speed-up Module**：渲染低维特征（如 128 维）而非全维度（如 512 维 LSeg），再用轻量 1x1 卷积上采样到教师维度——显著加速且精度损失小。
3. **统一损失**：L = L_rgb + γ * L_feature (L1 loss between rendered and teacher feature maps)，无需额外结构。

## 关键数字

| 指标 | 值 |
|------|-----|
| Replica mIoU | 0.787 (full) / 0.782 (speed-up) |
| vs NeRF-DFF mIoU | 0.787 vs 0.636 |
| 语义渲染 FPS | 6.84 (full) / 14.55 (speed-up) |
| 训练加速 vs NeRF | 2.7x |

## 与本文的关系

**高度相关——"附加特征到 Gaussian"的基础设施参考**。Feature 3DGS 的 parallel N-dimensional rasterizer 概念可以直接应用于本项目：我们可以在 Gaussian 渲染时同时输出 occupancy_alpha、semantic_id、confidence 等多个通道。speed-up module 的设计（低维存储→渲染→上采样）也是本项目存储小维度语义 embedding 的参考模式。

## 可用性

- [x] 代码开源
- [x] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- Feature 3DGS 证明了 Gaussian rasterizer 可以高效渲染多维特征——这对本项目的 multi-field occupancy representation（RGB + occupancy + semantic + confidence）至关重要。
- 与 LangSplat 和 LEGaussians 不同，Feature 3DGS 的特征蒸馏方式更直接（不需要 autoencoder/codebook），这与本项目 Phase 1 "简单直接"的工程追求一致。
- 项目的 speed-up module 给出了一个实用设计模式：在 Gaussian 上存储 compact features，在渲染后解码——这也是本项目应该采用的模式。
