# HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting

- **作者**：Jingyu Lin, Jiaqi Gu, Lubin Fan, Bojian Wu, Yujing Lou, Renjie Chen, Ligang Liu, Jieping Ye
- **会议/期刊**：CVPR 2025
- **年份**：2025
- **链接**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_HybridGS_Decoupling_Transients_and_Statics_with_2D_and_3D_Gaussian_CVPR_2025_paper.html)
- **代码**：[Project Page](https://gujiaqivadin.github.io/hybridgs/)
- **阅读日期**：2025-06-16

## 一句话总结

首次用 2D + 3D 高斯混合表示分离场景的静态和动态元素：3D Gaussians 建模多视角一致的静态场景（建筑物、地面等），2D Gaussians（per-image planar disks）建模只在单帧出现的 transient occluders（行人、车辆等），实现干净的场景分解。

## 核心方法

1. **静态用 3DGS、动态用 2DGS**：transient 物体缺乏多视角一致性（只在个别视角出现），所以用二维平面表示（per-image），自然将它们与三维一致的静态背景分离。
2. **多视角正则化监督**：利用 co-visible regions（多帧共视区域）增强静态/动态区分。
3. **多阶段训练**：warm-up → 交替训练 2DGS/3DGS → 联合微调。

## 关键数字

| 指标 | 值 |
|------|-----|
| NeRF On-the-go PSNR 提升 | ~1.1 dB vs prior methods |
| RobustNeRF (150 distractors) | 优秀 |
| 训练时间 (单 RTX 4090) | ~0.18 GPU hours |

## 与本文的关系

间接相关。HybridGS 的 2D/3D 分离思想对本项目处理动态场景有启发：我们的 static/dynamic decomposition 由本地安全回路处理（不在 scope 内），但 HybridGS 的静态-动态分离模式在未来扩展中可能有用——例如从多帧中分离出静止的障碍物（被观测多帧的纸箱）与瞬态干扰（单帧出现的行人）。

## 可用性

- [ ] 代码开源（待确认）
- [ ] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受（RTX 4090）

## 笔记

- 核心设计哲学对本项目有帮助："不同性质的场景元素用不同的 3D 表示"——本项目同样采用 dual representation（前景 Gaussian occupancy + 背景 pointmap），而不是用一个统一表示覆盖一切。
- 目前阶段本项目不做 dynamic scene，但 HybridGS 提供了一个清晰的"如果将来需要处理动态元素"的技术路线。
- 训练效率极高（0.18 GPU hours on RTX 4090），说明 hybrid approach 本身并不显著增加计算开销。
