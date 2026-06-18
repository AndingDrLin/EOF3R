# MASt3R & MUSt3R: From Image Matching to Multi-View 3D Reconstruction

- **作者**：MASt3R — Vincent Leroy, Yohann Cabon, Jérôme Revaud (Naver Labs Europe); MUSt3R — Yohann Cabon, Lucas Stoffl, Leonid Antsfeld et al.
- **会议/期刊**：MASt3R — ECCV 2024; MUSt3R — CVPR 2025
- **年份**：2024 / 2025
- **链接**：[MASt3R](https://arxiv.org/abs/2406.09756) | [MUSt3R](https://arxiv.org/abs/2503.01661)
- **代码**：MASt3R 集成在 DUSt3R 仓库; MUSt3R — [github.com/naver/must3r](https://github.com/naver/must3r)
- **阅读日期**：2025-06-16

## 一句话总结

MASt3R 在 DUSt3R 基础上增加 dense local feature matching head，将 3D 重建与特征匹配统一；MUSt3R 进一步将 DUSt3R 从图对扩展到多帧同时处理（对称架构+多层记忆机制），消除二次配对齐的瓶颈。

## 核心方法

1. **MASt3R（ECCV 2024）**：为 DUSt3R 网络增加一个额外的 head 输出 dense local features，配合 matching loss 训练和 fast reciprocal matching 方案，实现"grounded in 3D"的图像匹配——匹配已经在 3D 空间中完成，不依赖 2D 描述子。
2. **MUSt3R（CVPR 2025）**：对称架构直接预测所有视图的 3D 结构在统一坐标框架下，引入多层记忆机制可扩展到数千张 pointmap，同时支持 offline SfM 和 online SLAM 两种模式。

## 关键数字

| 指标 | 值 |
|------|-----|
| MASt3R 匹配精度 | 超过 RoMa（SOTA matcher） |
| MUSt3R 多视图重建 | SOTA（offline SfM + online SLAM） |
| MASt3R-SLAM 速度 | 15 FPS（单目密集 SLAM） |

## 与本文的关系

MASt3R/MUSt3R 是本项目背景模块的**扩展选项**。如果后续需要在线/多帧背景几何估计（例如 Husky 在移动中连续更新背景点云），MUSt3R 的多帧能力和 MASt3R-SLAM 的实时 SLAM 是潜在参考。当前阶段 DUSt3R/VGGT 已足够。

## 可用性

- [x] 代码开源
- [x] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- MASt3R 和 MUSt3R 的关系：MASt3R 增强图对匹配质量，MUSt3R 解决多帧效率问题。两者都是 DUSt3R 生态的不同进化分支。
- 本项目当前阶段不需要像素级匹配或多帧 SLAM 能力——Stage 4 只需要从 2-4 张场景图估计背景几何。但如果 Stage 7 实验中发现需要"连续帧背景更新"能力，MUSt3R/MASt3R-SLAM 提供了清晰的升级路径。
- MASt3R-SLAM 的 15 FPS 实时 SLAM 表明这类 feedforward 方法已经可以用于在线场景，这为本项目的车端异步架构提供了信心（云端即使 >1s 延迟也无妨，车端始终有本地回路）。
