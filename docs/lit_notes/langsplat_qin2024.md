# LangSplat: 3D Language Gaussian Splatting

- **作者**：Minghan Qin*, Wanhua Li*+, Jiawei Zhou*, Haoqian Wang+, Hanspeter Pfister
  (* equal contribution, + corresponding; Tsinghua & Harvard)
- **会议/期刊**：CVPR 2024 (Highlight)
- **年份**：2024
- **链接**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.html)
- **代码**：[github.com/minghanqin/LangSplat](https://github.com/minghanqin/LangSplat)
- **阅读日期**：2025-06-16

## 一句话总结

将 CLIP 语义特征附加到 3D Gaussians 上，通过 scene-wise autoencoder 将高维 CLIP 特征压缩为低维 latent code，利用 SAM 的分层 mask 学习多粒度语义，实现比 LERF（NeRF-based）快 199 倍的开放词汇 3D 语义查询。

## 核心方法

1. **Scene-wise Language Autoencoder**：为每个场景训练一个轻量 autoencoder，将 CLIP 的 512 维特征压缩到 3 维 latent code，大幅降低每个 Gaussian 的语义存储开销（~170x 压缩）。
2. **SAM 分层语义学习**：用 SAM 的三级 mask hierarchy（whole/part/subpart）训练多粒度语义，无需 DINO 特征正则化。
3. **Tile-based Semantic Splatting**：语义特征与 RGB 一同通过 Gaussian rasterization 渲染，查询时解码为 CLIP 空间做 cosine similarity 匹配。

## 关键数字

| 指标 | 值 |
|------|-----|
| vs LERF 速度 | 199x faster (1440x1080) |
| 3D-OVS mIoU | 93.4 |
| LERF dataset mIoU | 51.4 |
| 训练时间 | ~1.5 hrs |
| 训练显存 | ~6.2 GB |

## 与本文的关系

**间接相关——语义附加机制参考**。本项目 Phase 1 不需要开放词汇查询（使用预定义类别），但 LangSplat 的 autoencoder 压缩 + semantic splatting 模式启发了我们如何将语义信息附加到 Gaussian 上。我们的需求更简单：每个 Gaussian 只需存储 class ID（8-16 维 embedding）和 risk_score（标量），不需要 CLIP-level 的 512 维特征，因此可以在不压缩的情况下直接存储。

## 可用性

- [x] 代码开源
- [x] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- LangSplat 是 CVPR 2024 Highlight，是 semantic 3DGS 方向最有影响力的工作之一。后续有 LangSplat V2 (NeurIPS 2025, 450+ FPS) 和 4D LangSplat (CVPR 2025)。
- 本项目简化场景：不训练 autoencoder，直接存储小维度语义特征。在 Phase 2+ 如果需要开放词汇能力，LangSplat 的 autoencoder 压缩方案可以直接采用。
- 一个关键设计选择：LangSplat 的语义特征是通过 rasterization α-blended 渲染后查询的（像素级操作），而本项目需要在 3D 空间直接查询 Gaussian 的语义（用于 BEV 投影和占据分类）。这个差异意味着我们不需要 2D rendering-based 语义查询。
