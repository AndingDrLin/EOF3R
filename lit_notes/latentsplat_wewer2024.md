# latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction

- **作者**：Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, Jan Eric Lenssen
- **会议/期刊**：ECCV 2024
- **年份**：2024
- **链接**：[arXiv:2403.16292](https://arxiv.org/abs/2403.16292)
- **代码**：[github.com/Chrixtar/latentsplat](https://github.com/Chrixtar/latentsplat) (MIT License)
- **阅读日期**：2025-06-16

## 一句话总结

在 latent space 中预测 uncertainty-aware 的变分 3D Gaussians（每个 Gaussian 存储 SH coefficient 的 mean + variance），splat 后用轻量 VAE-GAN decoder 生成最终图像，在少观测区域显式量化不确定性，外推能力远超回归方法。

## 核心方法

1. **Variational 3D Gaussians**：每个 Gaussian 存储 SH coefficients 的分布（μ 和 σ），而非确定性的值。未被充分观测的 3D 位置天然获得高方差，实现 principled uncertainty modeling。
2. **Latent Gaussian Space**：在低维 latent space 中预测 Gaussians（而非 pixel space），splatting 后通过 VAE-GAN decoder 生成高细节图像，避免回归方法在 unseen regions 的模糊问题。
3. **纯视频数据训练**：无 3D 监督，仅用 RealEstate10K 和 CO3D 的真实视频训练。

## 关键数字

| 指标 | 值 |
|------|-----|
| RealEstate10K FID | 2.22 (vs pixelSplat: 4.41) |
| CO3D Chamfer Distance (Hydrants) | 1.535 × 10^-8 (vs pixelSplat: 1.815) |
| 编码时间 | 80 ms |
| 渲染时间 | 3 ms |
| 推理显存 | 3.16 GB |

## 与本文的关系

**架构借鉴价值**。latentSplat 的 variational uncertainty 概念与本项目的 "confidence" 字段高度对应——我们可以为每个 Gaussian 在预测 occupancy_alpha 的同时预测一个 uncertainty/confidence，用于 BEV 投影的加权融合。但 VAE-GAN decoder 不符合本项目需求（生成式模型可能产生幻觉，不适合安全关键的规划场景）。MIT License 比 pixelSplat 更友好。

## 可用性

- [x] 代码开源 (MIT)
- [x] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受（3.16 GB）

## 笔记

- latentSplat 的 variational formulation 是对前馈 3DGS 不确定性建模的关键贡献——机器人场景中，物体的某些面可能被遮挡或观测不足，confidence 信息对于规划安全至关重要（例如：high confidence 的占据面→安全绕行；low confidence 的占据面→更保守的距离）。
- 但 latentSplat 的 generative decoder 与我们"不追求逼真渲染"的定位冲突。我们的核心输出是几何（Gaussian center/scale/rotation + occupancy_alpha + confidence），不需要 latent space + VAE-GAN。
- 可以考虑的迁移方案：在 MVSplat 的 Gaussian 输出头中增加一个 confidence prediction 分支（类似 latentSplat 的 σ），用 photometric uncertainty 或 depth uncertainty 监督。
