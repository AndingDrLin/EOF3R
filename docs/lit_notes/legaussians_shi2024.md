# LEGaussians: Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding

- **作者**：Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, Shao-Hua Guan
- **会议/期刊**：CVPR 2024
- **年份**：2024
- **链接**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Language_Embedded_3D_Gaussians_for_Open-Vocabulary_Scene_Understanding_CVPR_2024_paper.html)
- **代码**：[Project Page](https://buaavrcg.github.io/LEGaussians/)
- **阅读日期**：2025-06-16

## 一句话总结

通过 feature quantization（将高维 CLIP/DINO 特征量化为紧凑 code）+ adaptive spatial smoothing（per-Gaussian learned uncertainty 降低语义空间频率），在保持实时渲染的同时实现开放词汇 3D 查询，但室外无界场景性能较弱。

## 核心方法

1. **Feature Quantization Scheme**：用 learnable discrete codebook 将高维语义特征量化为紧凑 code（每 Gaussian 存 8 维），大幅降低显存。
2. **Adaptive Spatial Smoothing**：每个 Gaussian 学习一个 uncertainty，用于降低语义特征的空间频率——高频语义特征在多视角下不一致，smoothing 解决了这个矛盾。
3. **实时查询**：渲染特征图后用轻量 MLP 解码为 codebook distribution，与文本 embedding 比较做开放词汇查询。

## 关键数字

| 指标 | 值 |
|------|-----|
| 3D-OVS mIoU | 88.5 |
| LERF dataset mIoU | 46.9 |
| Mip-NeRF360 mIoU | 29.1 (弱) |
| 查询时间 (LERF) | 36.7 ms |
| 训练显存 | 11 GB |
| 训练时间 | 1.3 hrs |

## 与本文的关系

参考价值中等。LEGaussians 的 quantization 思路优雅但对本项目 Phase 1 过度复杂（预定义类别不需要 codebook）。值得关注的是 **spatial smoothing with uncertainty**——这个机制解决了多视角语义不一致的问题，本项目在跨视角 Gaussian 语义一致性方面可能面临类似挑战。室外场景性能弱（Mip-NeRF360 mIoU 仅 29.1）是一个重要警示。

## 可用性

- [x] 代码（仓库状态待确认）
- [ ] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受（11 GB）

## 笔记

- 室外无界场景的弱性能（mIoU 29.1 vs LangSplat 的 57.3）对本项目重要——我们的校园场景是室外无界的，LEGaussians 在这类场景上可能不适用。
- Spatial smoothing with learned uncertainty 是一个值得研究的技术点：在多视角观测中，不同视角的语义特征可能不一致（遮挡、视角变化导致），学习一个 per-Gaussian 的 smoothing 强度可以帮助稳定语义标签。
