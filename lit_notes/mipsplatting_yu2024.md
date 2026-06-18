# Mip-Splatting: Alias-free 3D Gaussian Splatting

- **作者**：Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, Andreas Geiger
- **会议/期刊**：CVPR 2024 (Best Student Paper Award)
- **年份**：2024
- **链接**：[CVF Open Access](https://www.openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html)
- **代码**：[github.com/autonomousvision/mip-splatting](https://github.com/autonomousvision/mip-splatting)
- **阅读日期**：2025-06-16

## 一句话总结

指出现有 3DGS 在变尺度渲染时的混叠问题根源（缺乏 3D 频率约束 + 固定 2D 膨胀滤波器），提出 3D 平滑滤波器（按 Nyquist 频率限制 Gaussian 频率）和 2D Mip 滤波器（近似成像过程的 box filter）消除混叠。

## 核心方法

1. **3D 平滑滤波器**：根据每个训练视角的焦距、深度、像素采样间隔，推导每个 Gaussian 的最大采样频率，施加 3D 高斯低通滤波，使其频率保持在 Nyquist 极限以下。该滤波成为场景表示的固有部分。
2. **2D Mip 滤波器**：替换原始 3DGS 的简单 2D 膨胀滤波器，用高效 2D 高斯低通滤波近似物理成像过程的 box filter（像素面积积分的投影），消除缩小时的膨胀伪影。

## 关键数字

| 指标 | 值 |
|------|-----|
| 单尺度训练+多尺度测试 | 显著超过 Mip-NeRF 360、Zip-NeRF、3DGS |
| 标准单尺度训练+测试 | 与 3DGS 持平（无退化） |
| 渲染开销增加 | 0（3D 平滑滤波器可融合进 Gaussian primitive） |
| 训练开销 | 略增（每 100 次迭代需重新计算采样率） |

## 与本文的关系

本项目的机器人场景中，相机与物体的距离会动态变化（车靠近障碍物→物体变大；车远离→物体变小）。Mip-Splatting 的抗混叠能力使 Gaussian 在不同观测距离下保持几何一致，这对占据估计的准确性有帮助。但本项目当前阶段不追求多尺度渲染质量，**几何精度是主要关注点**。

## 可用性

- [x] 代码开源
- [ ] 权重可下载（训练模型需自行训练）
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- 获 CVPR 2024 Best Student Paper Award，是 3DGS 质量改进方向的代表作。
- 3D 平滑滤波器的核心价值：训练后的 Gaussian 表示在任何尺度下都不会出现高频伪影。这对几何提取有帮助——不会因为相机距离变化而产生虚假的 Gaussian 膨胀。
- 与我们的机器人场景关联有限（主要是渲染质量改进），但在多视角变距离观测（如 Husky 接近障碍物过程）中有潜在价值。
