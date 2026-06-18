# DUSt3R: Geometric 3D Vision Made Easy

- **作者**：Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jérôme Revaud
- **会议/期刊**：CVPR 2024
- **年份**：2024
- **链接**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html)
- **代码**：[github.com/naver/dust3r](https://github.com/naver/dust3r)
- **阅读日期**：2025-06-16

## 一句话总结

开创性地将传统多阶段 3D 重建 pipeline（特征提取→匹配→SfM→MVS）简化为端到端回归：从两张无标定图像直接回归对齐后的 3D pointmap，统一处理深度估计、位姿估计、密集重建等多个任务。

## 核心方法

1. **Pointmap 回归范式**：用 Siamese ViT 编码器 + Cross-Attention Transformer Decoder 从图对直接回归每个像素的 3D 坐标（在第一张图的坐标系下），配以逐像素置信度。
2. **Cross-Attention 是关键**：两个 decoder 分支共享信息（self-attention 处理同帧 → cross-attention 跨帧交换），确保两个 pointmap 在统一的坐标框架下。
3. **简单但高效的全局对齐**：对于 >2 张图，用 pointmap-based 全局对齐过程（类似 BA 但直接操作 3D 坐标），收敛快且鲁棒。
4. **训练**：8.5M 图对、8 个数据集混合，纯回归损失（confidence-weighted Euclidean loss）。

## 关键数字

| 指标 | 值 |
|------|-----|
| 单目深度 | SOTA（发表时） |
| 多视图深度 | SOTA |
| 相对位姿估计 | SOTA |
| 3 帧多视图位姿 (RRA@15/RTA@15/mAA30) | 95.3/88.3/77.5 |
| 推理速度（图对） | ~0.2s |
| 引用数 | 363+ |

## 与本文的关系

DUSt3R 是本项目背景模块的 **Fallback 方案**。如果 VGGT 部署遇到困难，DUSt3R 是成熟可靠的替代——代码完善、社区活跃、被大量下游工作引用。输入 2 张场景图即可获得 pointmap + 相对位姿。DUSt3R 的 pointmap 输出天然支持我们的坐标对齐和 BEV 投影需求。

## 可用性

- [x] 代码开源
- [x] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- DUSt3R 是"范式转变"之作——在此之前 3D 重建 = SfM + MVS 的固定 pipeline，DUSt3R 证明了单次前向传播就能做得更好。
- 它催生了一个庞大的后续工作生态（MASt3R、Spann3R、Splatt3R、MASt3R-SLAM 等），项目 README 中有两个 GitHub Awesome list 可以参考。
- 本项目使用场景：全场景图（或背景-masked 图）→ DUSt3R → pointmap + 位姿 → 送入 Stage 4 背景模块。如果只用 2 张图，不需要多视图全局对齐也能得到有意义的输出。
- DUSt3R 的一个局限是主要处理图对，多视图需要额外的对齐过程。这是 VGGT 相比它的主要提升点。
