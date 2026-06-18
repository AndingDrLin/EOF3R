# pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction

- **作者**：David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, Vincent Sitzmann
- **会议/期刊**：CVPR 2024 (Oral, Best Paper Runner-Up)
- **年份**：2024
- **链接**：[CVF Open Access](https://www.openaccess.thecvf.com/content/CVPR2024/html/Charatan_pixelSplat_3D_Gaussian_Splats_from_Image_Pairs_for_Scalable_Generalizable_CVPR_2024_paper.html)
- **代码**：[github.com/dcharatan/pixelsplat](https://github.com/dcharatan/pixelsplat)
- **阅读日期**：2025-06-16

## 一句话总结

首次实现前馈 3DGS：用 epipolar transformer 沿极线聚合跨视图特征隐式推理 3D 几何，通过概率化深度采样（可微重参数化）从 image pairs 预测 3D Gaussian primitives，无需测试时优化。

## 核心方法

1. **Epipolar Transformer**：对每个像素的射线，沿另一视图的极线采样特征（配以深度位置编码），通过 cross-attention 发现对应关系，self-attention 传播深度估计到无匹配区域。
2. **Probabilistic Gaussian Depth Sampling**：预测深度上的离散概率分布，通过可微重参数化采样深度，用采样概率作为 Gaussian 的 opacity——解决了直接回归 Gaussian center 的局部最优问题。
3. **Pixel-Aligned Gaussians**：每个输入像素对应一个 3D Gaussian，场景表示为各视图 Gaussian 的并集。

## 关键数字

| 指标 | 值 |
|------|-----|
| RealEstate10K PSNR/SSIM/LPIPS | 26.09/0.863/0.136 |
| ACID PSNR/SSIM/LPIPS | 28.27/0.843/0.146 |
| 训练显存 | 80 GB (A100/H100) |
| 推理速度 | <10 FPS |

## 与本文的关系

前馈 3DGS 的开山之作，学术界意义重大。但直接用于本项目前景模块有困难：显存需求太高（训练需要 80GB A100）、推理速度慢（<10 FPS）、跨数据集泛化弱于 MVSplat。epipolar transformer 架构和概率化采样思想值得在本项目的 G2O-inspired decoder 设计中借鉴。

## 可用性

- [x] 代码开源
- [x] 权重可下载
- [ ] 已在本地跑通
- [ ] 显存要求过高（80GB 训练）

## 笔记

- pixelSplat 是前馈 3DGS 的奠基之作（CVPR 2024 Oral + Best Paper Runner-Up），但已被 MVSplat 在速度、效率、几何精度、泛化能力上全面超越。
- 概率化深度采样的思想值得保留：本项目的前馈 decoder 也可能面临深度预测的局部最优问题，采样策略（而非确定性回归）可能带来更好的鲁棒性。
- pixelSplat 的代码被 latentSplat 等后续工作继承和改进，有一定的社区基础设施价值。
