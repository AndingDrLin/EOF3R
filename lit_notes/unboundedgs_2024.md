# Unbounded-GS: Extending 3DGS with Hybrid Representation for Unbounded Large-Scale Scene Reconstruction

- **作者**：（检索来源：IEEE RA-L 论文）
- **会议/期刊**：IEEE Robotics and Automation Letters (RA-L), Vol. 9, No. 12
- **年份**：2024
- **链接**：[IEEE Xplore](https://ieeexplore.ieee.org/document/10747249)
- **代码**：待确认
- **阅读日期**：2025-06-16

## 一句话总结

针对 3DGS 在无界大场景中远处背景高斯点覆盖不足导致的"斑驳"伪影，提出混合显式+隐式表示：前景用显式 3D Gaussians（高效 splatting），远处背景用 MLP 从位置编码预测颜色，两者通过无缝 blending 融合为完整场景。

## 核心方法

1. **混合显式+隐式表示**：前景（近场）区域用 3DGS splatting 保持效率；远处背景（远场）用轻量 MLP 从 view direction + 位置编码预测颜色，避免在高斯点密度不足的远距离区域产生伪影。
2. **无缝 Blending**：前景与背景的颜色通过混合权重融合，过渡平滑。
3. **避免采样空间膨胀**：不需要 NeRF 类方法中的场景包围盒采样（Mip-NeRF 360 等），规避了大规模场景中采样点数量爆炸的问题。

## 关键数字

| 指标 | 值 |
|------|-----|
| 收敛速度 | 快于 NeRF 类方法 |
| 渲染保真度 | 高于纯 3DGS（在远处背景） |
| 场景规模 | 适用于大范围无界场景 |

## 与本文的关系

**架构参考价值较高**。本项目同样面临"前景（近距离物体）+ 背景（远距离环境）"的混合表示需求，Unbounded-GS 的 hybrid explicit+implicit 思路直接映射到我们的架构：前景物体用 feedforward Gaussian occupancy 表示（类似 foreground Gaussians），背景用 VGGT/DUSt3R pointmap 表示（类似 background MLP/点云）。关键差异：我们不追求渲染融合，而是坐标对齐后投影到 BEV 占据网格。

## 可用性

- [ ] 代码开源（待确认）
- [ ] 权重可下载
- [ ] 已在本地跑通
- [ ] 显存要求待确认

## 笔记

- 发表在有 RA-L（机器人领域期刊）而非纯视觉会议，说明作者已经考虑了机器人应用场景——对本项目有间接背书。
- Hybrid 思路的核心启发：不需要用一个统一的表示处理所有区域。近处的物体需要精细几何（Gaussian occupancy），远处的环境只需要粗粒度的占用/通行性（pointmap → traversable mask）。
- 对于校园场景，远处背景往往是建筑物、树木等静态结构，VGGT 的点云图足以表示；前景才是规划关注的核心区域。
