# MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

- **作者**：Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, Jianfei Cai
- **会议/期刊**：ECCV 2024 (Oral)
- **年份**：2024
- **链接**：[arXiv:2403.14627](https://arxiv.org/abs/2403.14627) | [Project Page](https://donydchen.github.io/mvsplat)
- **代码**：[github.com/donydchen/mvsplat](https://github.com/donydchen/mvsplat)
- **阅读日期**：2025-06-16

## 一句话总结

通过 plane-sweep cost volume 从多视角图像构建 3D 特征体，然后用轻量 U-Net + cross-view attention 估计深度并预测 Gaussian 参数，实现 22 FPS 的前馈 3DGS，参数量仅为 pixelSplat 的 1/10，且跨数据集泛化能力更强。

## 核心方法

1. **Plane-sweep Cost Volume**：将多视角特征沿深度假设平面投影并计算相似度，构建 cost volume。这提供了强几何先验（不同于 pixelSplat 的 epipolar transformer 隐式几何推理），使 Gaussian center 定位更准确。
2. **Cross-View Attention Depth Refinement**：用 cross-view attention 融合多视角信息，细化深度估计。
3. **Joint Gaussian Parameter Prediction**：从 refined depth 出发，预测每个像素对应 Gaussian 的 opacity、covariance（scale+rotation）、SH coefficients。
4. **纯光度监督训练**：仅用渲染 RGB 与 GT RGB 的 L1+L2+SSIM 损失训练，无需 GT depth。

## 关键数字

| 指标 | 值 |
|------|-----|
| RealEstate10K PSNR/SSIM/LPIPS | 26.39/0.869/0.128 (vs pixelSplat: 25.89/0.818/0.181) |
| ACID PSNR/SSIM/LPIPS | 28.25/0.843/0.144 |
| 推理速度 | 22 FPS |
| 参数量 vs pixelSplat | ~10x fewer |
| 训练硬件 | 单 A100 |

## 与本文的关系

**本项目前景模块的首选基础架构**。选择 MVSplat 而非 pixelSplat 的核心原因：(1) cost volume 提供更强的几何约束——本项目需要准确的 Gaussian 位置来做占据估计，而非仅仅渲染好看；(2) 参数量少 10x，更适配 <12GB VRAM 目标；(3) 跨数据集泛化更好，对校园场景的零样本迁移更有利。本项目将扩展 MVSplat 的输出头：在 SH coefficients 之外增加 occupancy_alpha、semantic label、footprint 预测。

## 可用性

- [x] 代码开源
- [x] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- MVSplat 的 cost volume 方法与 DUSt3R/VGGT 的 cross-attention 回归形成了有趣的对比：cost volume 更显式地编码几何，理论上更适合几何精度要求高的任务——这正是本项目前馈占据预测需要的。
- 后续工作 DepthSplat (CVPR 2025) 支持最多 12 个视图，如果本项目的 2-4 视图不够用，有明确的升级路径。
- 消融实验表明 cost volume 是最关键的组件（去掉后 PSNR 从 26.39 降到 22.83），这进一步确认了 cost volume 对几何精度的贡献。
- 局限性：对反射/非朗伯特表面表现不佳——校园场景中金属路锥、玻璃门窗等需要注意。
