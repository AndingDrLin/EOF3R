# ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting

- **作者**：Ruijie Zhu, Mulin Yu, Linning Xu, Lihan Jiang, Yixuan Li, Tianzhu Zhang, Jiangmiao Pang, Bo Dai
- **会议/期刊**：ICCV 2025
- **年份**：2025
- **链接**：[arXiv:2507.15454](https://arxiv.org/abs/2507.15454)
- **代码**：[github.com/RuijieZhu94/ObjectGS](https://github.com/RuijieZhu94/ObjectGS)
- **阅读日期**：2025-06-16

## 一句话总结

在 3DGS 训练过程中引入 object-aware anchor（局部 anchor 生成神经 Gaussian 并共享 object ID），通过 SAM 生成跨视角一致的 2D mask 并多数投票赋 ID，最终将场景分解为带有离散语义 ID 的 per-object Gaussian 组，实现重建+分割的统一。

## 核心方法

1. **Object-aware Neural Gaussian Generation**：物体被建模为局部 anchor，每个 anchor 生成一组神经 Gaussian 并共享 object ID，anchor 在训练中动态增长和剪枝。
2. **Object ID Labeling & Voting**：用 SAM 在多个视角上生成 2D mask，通过多数投票机制为每个 Gaussian 分配稳定、跨视角一致的 object ID。
3. **Discrete Gaussian Semantics**：使用 fixed one-hot ID 编码（而非可学习的连续语义），消除 alpha blending 过程中的语义歧义，实现像素级物体识别。

## 关键数字

| 指标 | 值 |
|------|-----|
| 开放词汇分割 | SOTA |
| 全景分割 | SOTA |
| 渲染质量 | 与 3DGS 持平（无退化） |

## 与本文的关系

ObjectGS 是本项目 **"object-level" 路线** 的最直接参考之一。本项目的前景模块需要将每个物体独立表示为 Gaussian 组，ObjectGS 的 per-object anchor + ID labeling 思路可以直接借鉴。但 ObjectGS 仍基于逐场景优化（不是 feedforward），我们需要将其 object-separation 思想适配到 feedforward 架构中。

## 可用性

- [x] 代码开源
- [ ] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- ObjectGS 的 object ID 分配 pipeline（SAM → multi-view voting → per-Gaussian ID）是一个可以直接复用的工程方案。即使我们不逐场景优化 3DGS，ObjectGS 的 ID 传递逻辑仍然适用于：SAM2 分割 → 关联到 MVSplat feedforward 输出的 per-object Gaussians。
- 离散 one-hot ID 编码比连续语义 embedding 更适合我们的场景——我们只需要预定义的几个类别（行人、自行车、路锥等），不需要高维 CLIP feature。
- ObjectGS 的存在证明 object-level 3DGS 是可行的方向，可以减少我们对"object-level 分解是否可行"的技术风险。
