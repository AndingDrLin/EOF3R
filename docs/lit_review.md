# 文献调研

> 最后更新：2026-06-18 | Stage 1 完成

---

## 调研方向与阅读清单

### A. 3D Gaussian Splatting（基础）

要回答的问题：
- 3DGS 数学表达、高斯球参数、渲染过程？
- 训练/优化流程、需要什么输入？
- 已知局限性（背景、无界场景）？

**已读（笔记完成）**：
- [x] Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
  - 代码：`github.com/graphdeco-inria/gaussian-splatting`
  - 笔记：`lit_notes/3dgs_kerbl2023.md`
  - 对应本项目：前景物体占据形状估计的基础（理解 Gaussian primitive 的数学表达和渲染过程，但本项目不追求 photorealistic rendering）
- [x] Yu et al., "Mip-Splatting: Alias-free 3D Gaussian Splatting", CVPR 2024 (Best Student Paper)
  - 笔记：`lit_notes/mipsplatting_yu2024.md`
- [x] Huang et al., "2D Gaussian Splatting for Geometrically Accurate Radiance Fields", SIGGRAPH 2024
  - 笔记：`lit_notes/2dgs_huang2024.md`
  - 重要发现：2D surfels 的几何精度（Chamfer Distance）是 3DGS 的 2.4 倍提升，对本项目的占据边界提取有直接价值

### B. Object-level / Efficient 3DGS

要回答的问题：
- 现有方法如何分解场景（2D mask → 3D lift？直接在 3D 分解？）
- 如何处理物体边界和遮挡？
- 多物体的高斯球如何管理？

**已读（笔记完成）**：
- [x] Zhu et al., "ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting", ICCV 2025
  - 笔记：`lit_notes/objectgs_zhu2025.md`
  - 核心：object-aware anchor + discrete object ID + SAM 跨视角投票

**待读**：
- [ ] Gaussian Object Carver (GOC, arXiv 2412.02075) — object-compositional GS + surface completion
- [ ] Gaussian Grouping (Ye et al., ECCV 2024) — identity encoding per Gaussian
- [ ] GES: Generalized Exponential Splatting (CVPR 2024) — 更紧凑的 primitive 表示

### C. Feedforward 3D Reconstruction（3R 模型）

要回答的问题：
- DUSt3R → MASt3R → VGGT 演进路径？
- 各自的输入输出、速度、精度？
- 哪些模型开源且可复现？

**已读（笔记完成）**：
- [x] Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025 (Best Paper Award)
  - 笔记：`lit_notes/vggt_wang2025.md`
  - **背景模块首选**：纯前馈 Transformer，~1.2B 参数，0.14s/10frames，同时输出相机位姿+pointmap+深度+跟踪
- [x] Wang et al., "DUSt3R: Geometric 3D Vision Made Easy", CVPR 2024
  - 笔记：`lit_notes/dust3r_wang2024.md`
  - **背景模块 Fallback**：pointmap 回归范式开创者，成熟可靠
- [x] Leroy et al., "Grounding Image Matching in 3D with MASt3R", ECCV 2024 + MUSt3R (CVPR 2025)
  - 笔记：`lit_notes/mast3r_leroy2024.md`

**待读**：
- [ ] Zhang et al., "Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT", arXiv 2507.08448 — 综述

**参考资源**：
- [All-3R-SLAM-in-this-Repo](https://github.com/3D-Vision-World/All-3R-SLAM-in-this-Repo)
- [DUSt3R-Paper-Collection](https://github.com/minoTrey/DUSt3R-Paper-Collection)

### D. 场景分割（Segmentation / Detection）

要回答的问题：
- SAM2 vs YOLO 在精度/速度/类别先验需求上的对比？
- 如何将 2D mask 一致传播到多视图？
- 是否存在轻量替代方案？

**已读（笔记完成）**：
- [x] Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024
  - 笔记：`lit_notes/sam2_ravi2024.md`
  - **前景分割核心工具**：Tiny (91 FPS) 或 Small (85 FPS) 变体，视频 memory 机制跨帧跟踪
- [x] Cheng et al., "YOLO-World: Real-Time Open-Vocabulary Object Detection", CVPR 2024
  - 笔记：`lit_notes/yoloworld_cheng2024.md`
  - **语义检测候选**：52 FPS，与 SAM2 互补（提供 box + 类别 → SAM2 精细化 mask）

### E. Hybrid 3D Scene Representation

要回答的问题：
- 现有方法如何融合显式+隐式表示？
- 如何处理前景-背景接缝？
- 对齐策略是什么？

**已读（笔记完成）**：
- [x] Unbounded-GS (IEEE RA-L 2024) — foreground Gaussians + background MLP
  - 笔记：`lit_notes/unboundedgs_2024.md`
- [x] Lin et al., "HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting", CVPR 2025
  - 笔记：`lit_notes/hybridgs_lin2025.md`
  - 核心启发：3DGS 表示静态背景 + 2DGS 表示 transient 动态物体

### F. Autonomous Perception Demo

要回答的问题：
- 3D 重建在 perception pipeline 中的角色？
- demo 的最小可用形式是什么？
- 已有系统如何评估感知质量？

**已读（笔记完成）**：
- [x] Huang et al., "GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction", ECCV 2024
  - 笔记：`lit_notes/gaussianformer_huang2024.md`
  - **高度相关**：首次用 Gaussian 做占据预测（非渲染），证明了 Gaussian→Occupancy 路线的可行性

**待读**：
- [ ] EA3D (NeurIPS 2025) — 在线 3D 物体提取 + VLM 语义
- [ ] DrivingGaussian (CVPR 2024) — Composite Gaussian Splatting 用于自动驾驶

---

### G. Feedforward / Sparse-View 3DGS

对本项目的核心意义：传统 3DGS 需要 7000+ 次迭代训练，不适合"看几秒就要用"的机器人场景。Feedforward 3DGS 可以在单次前向传播中预测 Gaussian 参数。**本项目借鉴 G2O-GS 的几何约束思想（geometry scaffold、opacity confidence、edge supervision）但适配为 feedforward 模式。**

**已读（笔记完成）**：
- [x] Chen et al., "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images", ECCV 2024 (Oral)
  - 笔记：`lit_notes/mvsplat_chen2024.md`
  - **前景模块首选架构**：cost volume 强几何约束 + 22 FPS + 10x fewer params than pixelSplat
- [x] Charatan et al., "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction", CVPR 2024 (Oral)
  - 笔记：`lit_notes/pixelsplat_charatan2024.md`
  - 前馈 3DGS 开创者，但已被 MVSplat 在效率/几何/泛化上超越
- [x] Wewer et al., "latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction", ECCV 2024
  - 笔记：`lit_notes/latentsplat_wewer2024.md`
  - variational uncertainty 概念对本项目 confidence field 有直接借鉴价值

**待读**：
- [ ] Splatter-Image (CVPR 2024) — 单视图 3DGS
- [ ] Flash3DGS (2025) — 实时 feedforward

### H. Semantic 3DGS / Feature Distillation

对本项目的核心意义：每个前景物体的 Gaussians 需要继承语义信息。本项目 Phase 1 使用预定义类别，暂时不需要开放词汇。

**已读（笔记完成）**：
- [x] Qin et al., "LangSplat: 3D Language Gaussian Splatting", CVPR 2024 (Highlight)
  - 笔记：`lit_notes/langsplat_qin2024.md`
  - autoencoder 压缩 CLIP feature → Gaussian，对语义附加机制有参考价值
- [x] Shi et al., "LEGaussians: Language Embedded 3D Gaussians", CVPR 2024
  - 笔记：`lit_notes/legaussians_shi2024.md`
  - 量化 codebook + spatial smoothing，室外场景性能弱（Mip-NeRF360 mIoU 仅 29.1）
- [x] Zhou et al., "Feature 3DGS: Supercharging 3DGS to Enable Distilled Feature Fields", CVPR 2024
  - 笔记：`lit_notes/feature3dgs_zhou2024.md`
  - 并行 N 维光栅化器可直接用于本项目多字段 Gaussian 渲染

**待读**：
- [ ] OpenGaussian (NeurIPS 2024) — 点级开放词汇理解

### I. BEV Occupancy & Costmap Generation

**已读（笔记完成）**：
- [x] Tian et al., "Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark", NeurIPS 2023
  - 笔记：`lit_notes/occ3d_tian2023.md`
  - 占据预测任务定义（free/occupied/unobserved）+ 标注 pipeline 参考
- [x] Nav2 costmap_2d 官方文档
  - 笔记：`lit_notes/nav2_costmap2d_docs.md`
  - Layered Costmap 架构 + 自定义 plugin API，是本项目 costmap 模块的直接目标接口

**待读**：
- [ ] BEVDet / BEVFormer 系列 — BEV 特征生成（LSS / Transformer 两种范式）
- [ ] SurroundOcc (ICCV 2023)

### J. ROS2 Nav2 Local Planning

**已读（笔记完成）**：
- [x] Nav2 Controller/Planner 官方文档
  - 笔记：`lit_notes/nav2_planning_docs.md`
  - DWB（Dynamic Window Approach）是本项目的主要目标消费方，critic-based 架构通过采样 costmap 值评分轨迹

**待读**：
- [ ] DWB 原论文（ROS1 DWAPlanner）
- [ ] MPPI (Model Predictive Path Integral) 文档 — TEB 的推荐替代

### K. Edge-Cloud Robotics & Async Perception

**已读（笔记完成）**：
- [x] Chen et al., "FogROS2: An Adaptive Platform for Cloud and Fog Robotics Using ROS 2", IEEE ICRA 2023
  - 笔记：`lit_notes/fogros2_chen2023.md`
  - 架构参考（VPN + H.264 + Kubernetes），本项目不直接使用但借鉴设计模式

**待读**：
- [ ] FogROS2-Sky (ICRA 2024) — 多云端成本-延迟优化
- [ ] Cloud robotics latency characterization papers

### L. Campus / Last-Mile Delivery Perception

**已读（笔记完成）**：
- [x] Starship Technologies Delivery Robot — 公开技术文档分析
  - 笔记：`lit_notes/starship_delivery_tech.md`
  - 12 相机 + 雷达 + LiDAR + 超声波冗余方案，6.4 km/h L4 自主，>99% 自动驾驶率

**待读**：
- [ ] Nuro Driver 技术报告 — 多模态融合 + geometric fallback 层
- [ ] 美团无人配送 魔袋20 — 19 相机 + 3 LiDAR + 五层安全系统
- [ ] 京东物流 独狼 6 代 — Perception 5.0 端到端架构

---

## 关键发现与基线选择

### 各模块基线推荐

| 模块 | 首选方案 | Fallback | 选择理由 |
|------|----------|----------|----------|
| **前景分割** | YOLO-World (box + class) → SAM2 (mask refinement) | YOLOv8 + SAM2 | YOLO-World 提供语义类别，SAM2 提供精细 mask，两者互补 |
| **前景 Gaussian Occupancy** | MVSplat (cost volume + feedforward) | pixelSplat (epipolar transformer) | MVSplat 的 cost volume 提供更强几何约束（占据预测需要准确 3D 位置），22 FPS 前馈速度，10x 更少参数 |
| **背景几何估计** | VGGT (纯前馈，0.14s/10帧) | DUSt3R / MASt3R | VGGT 最快、最准、多任务统一输出，CVPR 2025 Best Paper。DUSt3R 是成熟 fallback |
| **语义附加** | Feature 3DGS 并行 N 维光栅化器 + 直接存储小维度 semantic embedding | LangSplat autoencoder（如需开放词汇） | Phase 1 预定义类别不需要 CLIP 压缩，直接存 class ID + confidence |
| **BEV 投影** | Gaussian-to-Voxel Splatting（参考 GaussianFormer） | 简单高度阈值 + 俯视投影 | GaussianFormer 已有 CUDA 加速实现 |
| **Costmap Plugin** | 自定义 Nav2 Layer Plugin (updateWithMax 合并) | — | 云端增强层叠加到本地 LiDAR 层，超时自动失活 |
| **车-云通信** | 自定义 HTTP/gRPC + H.264 压缩（参考 FogROS2） | FogROS2 完整框架 | 只需关键帧上传+costmap patch 下发，不需要完整 ROS2 节点卸载 |
| **局部规划** | Nav2 DWB (默认) + RPP (简洁备选) | MPPI | DWB 的 critic 架构天然适配多源 costmap 融合 |

### 前 10 篇优先阅读顺序

1. Zhang et al., "Review of Feed-forward 3D Reconstruction", arXiv 2025 — **先读，建框架**（待读）
2. Kerbl et al., "3D Gaussian Splatting", SIGGRAPH 2023 — 基础（已读）
3. Wang et al., "VGGT", CVPR 2025 — 背景重建首选（已读）
4. Chen et al., "MVSplat", ECCV 2024 — 前景重建首选（已读）
5. Ravi et al., "SAM 2", 2024 — 分割基础（已读）
6. Zhu et al., "ObjectGS", ICCV 2025 — object-level GS（已读）
7. Nav2 官方文档 — costmap_2d + local planner architecture（已读）
8. Qin et al., "LangSplat", CVPR 2024 — semantic 3DGS 参考（已读）
9. Tian et al., "Occ3D", NeurIPS 2023 — BEV 占据表示（已读）
10. Huang et al., "GaussianFormer", ECCV 2024 — Gaussian 占据预测（已读）

---

## 已读笔记清单（24 篇）

| # | 笔记文件 | 方向 | 论文 | 年份 | 会议 |
|---|----------|------|------|------|------|
| 1 | `3dgs_kerbl2023.md` | A | 3D Gaussian Splatting | 2023 | SIGGRAPH |
| 2 | `mipsplatting_yu2024.md` | A | Mip-Splatting | 2024 | CVPR |
| 3 | `2dgs_huang2024.md` | A | 2D Gaussian Splatting | 2024 | SIGGRAPH |
| 4 | `objectgs_zhu2025.md` | B | ObjectGS | 2025 | ICCV |
| 5 | `vggt_wang2025.md` | C | VGGT | 2025 | CVPR |
| 6 | `dust3r_wang2024.md` | C | DUSt3R | 2024 | CVPR |
| 7 | `mast3r_leroy2024.md` | C | MASt3R & MUSt3R | 2024/25 | ECCV/CVPR |
| 8 | `sam2_ravi2024.md` | D | SAM 2 | 2024 | arXiv |
| 9 | `yoloworld_cheng2024.md` | D | YOLO-World | 2024 | CVPR |
| 10 | `unboundedgs_2024.md` | E | Unbounded-GS | 2024 | IEEE RA-L |
| 11 | `hybridgs_lin2025.md` | E | HybridGS | 2025 | CVPR |
| 12 | `gaussianformer_huang2024.md` | F | GaussianFormer | 2024 | ECCV |
| 13 | `mvsplat_chen2024.md` | G | MVSplat | 2024 | ECCV |
| 14 | `pixelsplat_charatan2024.md` | G | pixelSplat | 2024 | CVPR |
| 15 | `latentsplat_wewer2024.md` | G | latentSplat | 2024 | ECCV |
| 16 | `langsplat_qin2024.md` | H | LangSplat | 2024 | CVPR |
| 17 | `legaussians_shi2024.md` | H | LEGaussians | 2024 | CVPR |
| 18 | `feature3dgs_zhou2024.md` | H | Feature 3DGS | 2024 | CVPR |
| 19 | `occ3d_tian2023.md` | I | Occ3D Benchmark | 2023 | NeurIPS |
| 20 | `nav2_costmap2d_docs.md` | I | Nav2 costmap_2d | — | 文档 |
| 21 | `nav2_planning_docs.md` | J | Nav2 Local Planning | — | 文档 |
| 22 | `fogros2_chen2023.md` | K | FogROS2 | 2023 | ICRA |
| 23 | `starship_delivery_tech.md` | L | Starship Delivery Robot | — | 产品分析 |

---

## 跨方向关键洞察

1. **Feedforward is the right paradigm**: VGGT (0.14s), MVSplat (22 FPS), PE3R (5 min/scene) 都证明了前馈方案在速度和精度上可行。逐场景优化（3DGS 需要 30K 迭代/40min）不适合机器人场景。

2. **几何优先于渲染**: 2DGS 的几何精度 2.4x 优于 3DGS，GaussianFormer 证明了 Gaussian 可以做占据（不只是渲染），Occ3D 定义了占据评估标准——这些共同指向本项目"规划导向 Gaussian occupancy"的定位是正确的。

3. **Object-level decomposition is validated**: ObjectGS 证明了 object-level 3DGS 可行；HybridGS 证明了 2D+3D 混合表示可以分离静态/动态——为本项目的 per-object foreground + background 架构提供了直接背书。

4. **Semantic attachment 已有成熟方案**: Feature 3DGS 的并行 N 维光栅化器解决了"附加特征到 Gaussian"的基础设施问题；我们只需存储小维度 embedding（class ID + confidence），不需要 LangSplat 的 CLIP 压缩。

5. **Nav2 layered costmap 天然适配异步增强**: 云端 costmap layer 可以作为独立 plugin 随时 activate/deactivate，超时自动失活——这是"本地安全回路 + 云端增强"架构的完美表达。

6. **真实部署验证了低速自主的可行性**: Starship (>99% L4, 1000 万次交付) 和 美团 (>99% L4, 500 万次交付) 证明了 campus/园区低速自主配送是技术上可行的——我们的感知方案建立在验证过的应用场景之上。
