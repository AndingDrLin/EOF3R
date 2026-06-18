# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

- **作者**：Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao
- **会议/期刊**：SIGGRAPH 2024
- **年份**：2024
- **链接**：[ACM DL](https://dl.acm.org/doi/fullHtml/10.1145/3641519.3657428) | [Project Page](https://surfsplatting.github.io)
- **代码**：[github.com/hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)
- **阅读日期**：2025-06-16

## 一句话总结

将 3DGS 的体素化 3D Gaussian 退化为 2D 定向平面圆盘（surfels），解决 3D Gaussian 多视角不一致导致的表面重建不准确问题，配合透视精确的 splatting 和深度畸变+法向一致性正则化，在保持实时渲染速度的同时显著提升几何精度。

## 核心方法

1. **2D Gaussian Primitives（surfels）**：用 2D 定向平面圆盘替代 3D 体素 Gaussian，给每个 primitive 一个明确的平面方位，本质上是"扁平化的 Gaussian"，几何上更接近真实表面。
2. **透视精确 Splatting**：用 ray-splat intersection 替代简单的投影近似，使深度渲染更准确。
3. **两个正则化损失**：深度畸变损失（让 2D primitives 沿每条光线紧密分布）+ 法向一致性损失（让渲染法向与深度梯度方向对齐），显著平滑表面。

## 关键数字

| 指标 | 值 |
|------|-----|
| 几何精度 | SOTA（显著优于 3DGS） |
| 渲染速度 | 实时（与 3DGS 同级） |
| 表面重构质量 | 大幅提升 |

## 与本文的关系

2DGS 的核心优势是**几何精度高于 3DGS**，这对本项目非常关键——我们需要准确的物体表面和占据边界来做 BEV footprint。2DGS 的平面圆盘 representation 理论上比 3DGS 的体素球更适合提取物体轮廓。但 2DGS 是否支持 feedforward 模式尚待确认（原始 2DGS 是逐场景优化的）。考虑作为前景 Gaussian 的类型选项之一（2D surfels vs 3D Gaussians for occupancy）。

## 可用性

- [x] 代码开源
- [ ] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- 2DGS 是目前 3DGS 几何精度改进方向的标杆方法，已被多个下游任务引用（如 Gaussian SLAM）。
- 本项目如果追求更准确的物体 surface/contour，2DGS 的 surfel 形式理论上比 3DGS 的 volumetric Gaussian 更合适——surfel 的边界更明确，BEV 投影的 footprint 更清晰。
- 但 surfel 的 feedforward 预测目前尚不成熟，这可能是 Stage 3 的一个备选方向（如果不满意 3DGS 的几何精度）。
- 与 3DGS 同组作者（Tübingen / Geiger lab），与 Mip-Splatting 有技术延续性。
