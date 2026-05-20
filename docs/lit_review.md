# 文献调研

## 调研方向与阅读清单

### A. 3D Gaussian Splatting（基础）

要回答的问题：
- 3DGS 数学表达、高斯球参数、渲染过程？
- 训练/优化流程、需要什么输入？
- 已知局限性（背景、无界场景）？

**必读**：
- [ ] Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
  - 代码：`github.com/graphdeco-inria/gaussian-splatting`
  - 对应本项目：前景物体占据形状估计的基础（理解 Gaussian primitive 的数学表达和渲染过程，但本项目不追求 photorealistic rendering）

**参考**：
- [ ] Mip-Splatting (CVPR 2024) — 抗锯齿
- [ ] Scaffold-GS (CVPR 2024) — 结构化 anchor
- [ ] 2D Gaussian Splatting (SIGGRAPH 2024) — 表面建模

### B. Object-level / Efficient 3DGS

要回答的问题：
- 现有方法如何分解场景（2D mask → 3D lift？直接在 3D 分解？）
- 如何处理物体边界和遮挡？
- 多物体的高斯球如何管理？

**阅读清单**：
- [ ] ObjectGS (Zhu et al., ICCV 2025) — object-aware anchors + semantic IDs
- [ ] Gaussian Object Carver (arXiv 2412.02075) — object-compositional GS + surface completion
- [ ] GS-Octree (CGF/Eurographics 2024) — octree SDF + GS
- [ ] GES (CVPR 2024) — Generalized Exponential Splatting

### C. Feedforward 3D Reconstruction（3R 模型）

要回答的问题：
- DUSt3R → MASt3R → VGGT 演进路径？
- 各自的输入输出、速度、精度？
- 哪些模型开源且可复现？

**必读**：
- [ ] Zhang et al., "Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT", arXiv 2507.08448 — **先读这篇**
- [ ] VGGT (Wang et al., CVPR 2025) — 代码：`github.com/facebookresearch/vggt`
- [ ] DUSt3R (CVPR 2024)
- [ ] MASt3R (CVPR 2025) — DUSt3R + feature matching

**参考资源**：
- [All-3R-SLAM-in-this-Repo](https://github.com/3D-Vision-World/All-3R-SLAM-in-this-Repo)
- [DUSt3R-Paper-Collection](https://github.com/minoTrey/DUSt3R-Paper-Collection)

### D. 场景分割（Segmentation / Detection）

要回答的问题：
- SAM2 vs YOLO 在精度/速度/类别先验需求上的对比？
- 如何将 2D mask 一致传播到多视图？
- 是否存在轻量替代方案？

**阅读/调研清单**：
- [ ] SAM2 官方文档（`github.com/facebookresearch/sam2`）
- [ ] YOLOv8/v11 + SAM2 集成（Ultralytics 文档）
- [ ] PE3R (NUS 2025) — SAM + DUSt3R → 3D 重建

### E. Hybrid 3D Scene Representation

要回答的问题：
- 现有方法如何融合显式+隐式表示？
- 如何处理前景-背景接缝？
- 对齐策略是什么？

**阅读清单**：
- [ ] Unbounded-GS (IEEE RA-L 2024) — foreground Gaussians + background MLP
- [ ] HybridGS (CVPR 2025) — 3DGS (static) + 2DGS (transient)
- [ ] Omni-Scene / Omni-Gaussian (CVPR 2025)

### F. Autonomous Perception Demo

要回答的问题：
- 3D 重建在 perception pipeline 中的角色？
- demo 的最小可用形式是什么？
- 已有系统如何评估感知质量？

**阅读/参考清单**：
- [ ] EA3D (Extract Anything 3D, arXiv 2510.25146) — 在线 3D 物体提取
- [ ] SAB3R (CVPR 2025 workshop) — "Map and Locate" task
- [ ] RAZER (arXiv 2505.15373) — open-vocabulary panoptic reconstruction
- [ ] LOCATE 3D (Meta FAIR, 2025) — 语言驱动的 3D 物体定位

---

### G. Feedforward / Sparse-View 3DGS

与本项目的关系：传统 3DGS 需要 7000+ 次迭代训练，不适合"看几秒就要用"的机器人场景。Feedforward 3DGS 可以在单次前向传播中预测 Gaussian 参数。**本项目借鉴 G2O-GS 的几何约束思想（geometry scaffold、opacity confidence、edge supervision）但适配为 feedforward 模式，不照搬逐场景优化流程。**

要回答的问题：
- 哪些 feedforward 3DGS 方法开源且可复现？
- 在 2-4 个视角输入下，几何精度能达到什么水平？
- G2O-GS 的 geometry-guided 思想如何转化为前馈架构的设计约束？
- 输出格式是否易转换为 BEV footprint？

**阅读清单**：
- [ ] **G2O-GS** (2025) — Geometry-guided Gaussian Occupancy: **核心思想来源**
  - 借鉴：geometry scaffold、opacity/confidence-aware selection、edge supervision、boundary-aware loss
  - 不照搬：逐场景 30k 迭代优化、clone/split densification、高阶 SH 渲染
- [ ] MVSplat (Chen et al., ECCV 2024) — cost-volume based, multiple dataset support
- [ ] pixelSplat (Charatan et al., CVPR 2024) — epipolar transformer
- [ ] latentSplat (Wysocki et al., CVPR 2025) — latent diffusion prior
- [ ] Splatter-Image (Szymanowicz et al., CVPR 2024) — single-view 3DGS prediction
- [ ] Flash3DGS (2025) — real-time feedforward 3DGS

**关注重点**：MVSplat（多视角、开源、精度较好）和 G2O-GS（几何约束思想借鉴）。G2O-GS 的 geometry-guided optimization 思想在本项目中转化为 geometry-guided feedforward occupancy prediction。

### H. Semantic 3DGS / Feature Distillation

与本项目的关系：每个前景物体的 Gaussians 需要继承语义信息（类别、实例 ID、风险等级）。主要方式有两种：feature field distillation（从 2D 特征图提升到 3D）和 per-Gaussian semantic label。

要回答的问题：
- 如何将 2D 语义特征高效地附加到 3DGS 上？
- 附加语义是否显著增加计算开销？
- 语义特征在少视角条件下是否仍然可靠？

**阅读清单**：
- [ ] LangSplat (Qin et al., CVPR 2024) — CLIP feature field in 3DGS
- [ ] LEGaussians (Shi et al., ECCV 2024) — language embedded 3DGS
- [ ] Feature-3DGS (Zhou et al., 2024) — feature field distillation
- [ ] OpenGaussian (Wu et al., NeurIPS 2024) — open-vocabulary 3DGS

**注意**：本项目第一阶段不要求 open-vocabulary。先用预定义类别（人、自行车、路锥、纸箱等），语义从 2D detector 继承到 3D Gaussian 组。Semantic feature field 是扩展方向，不是主线。

### I. BEV Occupancy & Costmap Generation

与本项目的关系：BEV 占据代价地图是连接 3D 感知和路径规划的桥梁。需要理解如何将 3D Gaussian 表示转换为 2D 占据网格。

要回答的问题：
- 从 3D 点云/Gaussians 到 BEV 占据网格的投影方法有哪些？
- 语义占据网格（semantic occupancy）的数据格式和标准？
- 如何生成 ROS2 Nav2 costmap_2d 兼容的 layer？
- 如何设置膨胀参数（inflation）使其既安全又不过度保守？

**阅读/调研清单**：
- [ ] BEVDet / BEVFormer 系列 — BEV 特征生成方法（参考，不一定直接用）
- [ ] Occ3D / OpenOccupancy — 3D 占据数据集和评估
- [ ] SurroundOcc (ICCV 2023) — 多相机 semantic occupancy
- [ ] Nav2 costmap_2d 官方文档 — layered costmap, inflation, obstacle/voxel layers
- [ ] Nav2 costmap plugin 开发文档 — 自定义 layer 的 API

**重点关注**：Nav2 costmap_2d 的 plugin 接口，以及如何将 3DGS 投影到 2D 网格（高度阈值过滤 + 俯视投影）。

### J. ROS2 Nav2 Local Planning

与本项目的关系：Nav2 局部规划器是本项目的"下游消费者"。云端生成的 costmap 最终要通过 Nav2 的 local planner 影响路径。

要回答的问题：
- Nav2 local planner（DWB, TEB, RPP）各自的原理和参数？
- 如何在 Nav2 中叠加自定义 costmap layer？
- Nav2 如何处理多 source 的 costmap（融合策略）？
- 局部规划器的速度限制和急停机制？

**阅读/调研清单**：
- [ ] Nav2 官方文档 — architecture overview, costmap_2d, planner/controller servers
- [ ] DWB (Dynamic Window Approach) 论文和 Nav2 实现
- [ ] TEB (Timed Elastic Band) 论文和 Nav2 实现
- [ ] Regulated Pure Pursuit (RPP) — Nav2 默认 controller
- [ ] ROS2 Navigation2 GitHub（`github.com/ros-navigation/navigation2`）

**重点关注**：Nav2 的 layered costmap 架构和自定义 layer plugin 的开发方式。

### K. Edge-Cloud Robotics & Async Perception

与本项目的关系：车端-云端异步架构是系统的核心设计。需要理解已有的云端机器人框架和延迟处理方法。

要回答的问题：
- 现有的云端机器人通信框架（FogROS2, Rapyuta）提供了什么？
- 延迟 >1s 的感知信息对机器人控制还有什么用？
- 如何设计"异步增强"而非"同步依赖"的系统？

**阅读/调研清单**：
- [ ] FogROS2 (Berkeley, 2023) — ROS2 cloud robotics framework
- [ ] Rapyuta (ETH Zurich) — cloud robotics platform
- [ ] ROS2 topic bridge over WAN — 跨网络 ROS2 通信方案
- [ ] Cloud robotics latency characterization papers — 延迟对控制的影响分析

**注意**：本项目不要求使用完整的 FogROS2/Rapyuta 框架（太重）。重点是理解异步通信的 design pattern，然后实现一个简单够用的版本。

### L. Campus / Last-Mile Delivery Perception

与本项目的关系：理解校园配送机器人的实际感知需求和已有系统的做法。

要回答的问题：
- 现有校园配送机器人（Starship, Nuro, 美团, 京东）使用什么感知方案？
- 它们的避障策略和局限性是什么？
- 低速近场场景的特殊挑战是什么？

**阅读/调研清单**：
- [ ] Starship Technologies — campus delivery robot 公开资料
- [ ] Nuro — last-mile autonomous delivery 技术报告
- [ ] 美团无人配送 — 国内校园/园区方案
- [ ] 京东物流机器人 — 园区配送感知方案

---

### M. Planning-Oriented 3D Representations

与本项目的关系：本项目不追求 photorealistic reconstruction，而是面向规划的几何-语义表示。需要调研已有工作中是否有类似"为规划而感知"的 3D 表示设计思路。

要回答的问题：
- 是否存在将 3D 表示直接用于规划的先行工作？
- 占据网格（occupancy grid）与 Gaussian 表示的融合方式？
- 如何评估"规划导向"的 3D 表示质量（非渲染指标）？

**阅读/调研清单**：
- [ ] G2O-GS (2025) — Geometry-guided Gaussian Occupancy，本项目的核心思想来源
- [ ] OccNeRF / OccNet 系列 — 以占据预测为目标的神经表示
- [ ] VoxFormer (CVPR 2023) — transformer-based 3D occupancy prediction
- [ ] GaussianFormer (2024) — Gaussian-based occupancy prediction
- [ ] DriveGaussian / ADGaussian — 自动驾驶场景的 Gaussian 表示

**重点关注**：以占据精度（而非渲染质量）为训练和评估目标的 3D 表示方法。G2O-GS 的 geometry-guided 思想（geometry scaffold, opacity confidence, edge supervision）是本项目的直接灵感来源。

---

## Baseline 选择（待调研后确认）

| 模块 | 优先方案 | Fallback | 状态 |
|------|----------|----------|------|
| 前景分割 | SAM2 (box prompt) | YOLOv8 + SAM2 | 待确定 |
| 前景占据估计 | MVSplat (feedforward) + G2O-inspired | 3DGS per-object optimization | 待确定 |
| 背景几何估计 | VGGT / MASt3R | DUSt3R | 待确定 |
| 融合与 BEV 投影 | 自行设计 | — | 待设计 |
| Costmap 生成 | Nav2 costmap_2d custom plugin | — | 待开发 |
| 车-云通信 | 自行设计（HTTP/gRPC + ROS2 topic） | FogROS2 参考 | 待设计 |
| 局部规划 | Nav2 RPP / DWB | — | 待配置 |
| 机器人平台 | Husky (ROS2 Humble) | Gazebo simulation | 待确定 |

---

## 前 10 篇优先阅读顺序

1. Zhang et al., "Review of Feed-forward 3D Reconstruction", arXiv 2025 — **先读，建框架**
2. Kerbl et al., "3D Gaussian Splatting", SIGGRAPH 2023 — 基础
3. Wang et al., "VGGT", CVPR 2025 — 背景重建首选
4. MVSplat (Chen et al., ECCV 2024) — 前景重建首选
5. Kirillov et al., "SAM 2", 2024 — 分割基础
6. ObjectGS (Zhu et al., ICCV 2025) — object-level GS
7. Nav2 官方文档 — costmap_2d + local planner
8. LangSplat (Qin et al., CVPR 2024) — semantic 3DGS 参考
9. BEVDet / Occ3D — BEV 占据表示（参考思路）
10. FogROS2 — 云端机器人通信（参考架构）
