# Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving

- **作者**：Xiaoyu Tian, Tao Jiang, Longfei Yun et al. (Tsinghua MARS Lab / USC / Shanghai AI Lab)
- **会议/期刊**：NeurIPS 2023 (Datasets & Benchmarks track)
- **年份**：2023
- **链接**：[arXiv:2304.14365](https://arxiv.org/abs/2304.14365) | [Project Page](https://tsinghua-mars-lab.github.io/Occ3D/)
- **代码**：[github.com/Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- **阅读日期**：2025-06-16

## 一句话总结

为自动驾驶 3D 占据预测建立了首个大规模基准数据集（Occ3D-nuScenes + Occ3D-Waymo），定义了占据预测任务（对每个 voxel 预测 free/occupied/unobserved 状态 + 语义标签），并提出了半自动标注 pipeline（LiDAR densification + occlusion reasoning + image-guided refinement）。

## 核心方法

1. **3D Occupancy Task Definition**：每个 voxel 预测三元状态（free / occupied / unobserved），occupied 的 voxel 还需预测语义类别（16 类 + General Object）。
2. **半自动标注 Pipeline**：多帧 LiDAR 聚合与 densification → occlusion reasoning（LiDAR + camera ray-casting 区分真正占据 vs 未观测）→ image-guided voxel refinement（消除传感器噪声和位姿误差导致的 3D-2D 不对齐）。
3. **两个数据集**：Occ3D-nuScenes (40K 帧, 6 cameras, 200x200x16 voxels @ 0.4m) 和 Occ3D-Waymo (200K 帧, 5 cameras, 高达 3200x3200x128)。

## 关键数字

| 指标 | 值 |
|------|-----|
| Occ3D-nuScenes 训练帧数 | ~40,000 |
| Occ3D-Waymo 训练帧数 | ~200,000 |
| 空间范围 (nuScenes) | [-40m, -40m, -1m, 40m, 40m, 5.4m] |
| 类别数 | 16 known + 1 GO |
| SOTA mIoU (nuScenes, EFFOcc) | 51.49 |

## 与本文的关系

**任务定义参考**。虽然本项目不是自动驾驶场景，但 Occ3D 定义的"占据预测"任务范式对我们设计 BEV occupancy grid 有价值：(1) 对每个 cell 预测 free/occupied/unknown 三元状态；(2) occupied cell 预测语义标签。这个格式可以直接适配到 ROS2 costmap 的 cost value（free→0, occupied→254, unknown→255）。主要差异：我们的 voxel 更小更近（近场场景），类别集合更精简。

## 可用性

- [x] 代码开源
- [x] 数据集可下载
- [ ] 已在本地跑通
- [x] 数据量可接受（预处理后）

## 笔记

- Occ3D 的最大贡献是建立了"3D 占据预测"这一自动驾驶子任务的 benchmark 和标准——在此之前，行业主要依赖 3D bounding box detection，无法处理不规则形状和 OOD 物体。
- 标注 pipeline 的 occlusion reasoning 步骤（区分 occupied vs unobserved voxels）对本项目有启发——我们的 Gaussian→BEV 投影也需要区分"确实有 Gaussian 覆盖的区域"和"没有观测到的盲区"。
- "General Object" (GO) 类别是一个聪明的设计：预留一个类别给训练集未出现的新物体。本项目可以借鉴——对所有"看起来是障碍物但不知道具体类别"的物体标记为 GO，赋予中性风险值。
