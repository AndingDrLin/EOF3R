# SAM 2: Segment Anything in Images and Videos

- **作者**：Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr et al. (Meta FAIR)
- **会议/期刊**：arXiv (Meta AI 发布)
- **年份**：2024
- **链接**：[arXiv:2408.00714](https://arxiv.org/abs/2408.00714) | [Project Page](https://ai.meta.com/sam2/)
- **代码**：[github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **阅读日期**：2025-06-16

## 一句话总结

将 SAM 的图像分割能力扩展到视频领域（图像视为单帧视频），通过 streaming memory 机制维持跨帧的对象记忆，实现 promptable visual segmentation（点/框/mask prompt → 全帧 masklet），比 SAM1 快 6 倍且更准。

## 核心方法

1. **Streaming Memory 架构**：Hiera 图像编码器（MAE-pretrained）+ Memory Attention（self-attention + cross-attention to memory bank）+ Memory Encoder（逐帧生成"记忆"存入 memory bank），帧按序处理但支持来自未来的 prompt。
2. **Memory Bank**：存储过去帧的特征和 mask 预测，通过 cross-attention 影响当前帧的分割。在遮挡后重新出现时，可通过一次点击恢复对象。
3. **Prompt Encoder + Mask Decoder**：与 SAM1 相同（点/框/mask prompt），增加 occlusion head 判断对象是否在当前帧可见。
4. **SA-V 数据集**：50.9K 视频、35.5M masks，是之前最大视频分割数据集的 53 倍。

## 关键数字

| 指标 | 值 |
|------|-----|
| vs SAM1 图像分割速度 | 6x faster |
| vs SAM1 图像分割精度 | 更高 |
| 视频分割交互次数 vs 先前方法 | 3x fewer |
| 标注效率 vs 先前 model-assisted 方法 | 8.4x faster |
| 零样本图像分割基准 | 37 个 |

## 与本文的关系

**本项目前景分割模块的核心工具**。SAM2 负责将场景分解为 per-object 2D mask（box prompt 或 automatic mode），然后将 mask 和 object crop 传给前景 MVSplat/3DGS。视频 segmentation 能力对多帧跟踪非常有用——Husky 观测到同一个物体多帧后，SAM2 的 memory 机制可以跨帧一致地分割同一物体。

## 可用性

- [x] 代码开源（Apache 2.0）
- [x] 权重可下载（SAM 2.1）
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- SAM2 的视频能力是本项目的关键资产——Husky 会从不同角度多次观测同一物体，跨帧一致分割对 per-object 3D 重建至关重要。如果在每帧独立用 SAM1 分割，同一物体可能被识别为不同 ID；SAM2 的 memory 机制解决了这个问题。
- 本项目的分割 pipeline 可能设计为：YOLO-World 提供 box prompt（带类别）→ SAM2 基于 box prompt 生成精细 mask → memory 跨帧跟踪。
- SA-V 数据集虽然不直接适用于本项目，但 SAM2 在零样本上的表现表明它可以直接在校园场景中使用（不需要微调）。
