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
  - 对应本项目：前景物体精细重建的基础

### B. Object-level / Efficient 3DGS

要回答的问题：
- 现有方法如何分解场景（2D mask → 3D lift？直接在 3D 分解？）
- 如何处理物体边界和遮挡？
- 多物体的高斯球如何管理？

**阅读清单**：
- [ ] ObjectGS (Zhu et al., ICCV 2025) — object-aware anchors + semantic IDs
- [ ] Gaussian Object Carver (arXiv 2412.02075) — object-compositional GS + surface completion
- [ ] GS-Octree (CGF/Eurographics 2024) — octree SDF + GS

### C. Feedforward 3D Reconstruction

要回答的问题：
- DUSt3R → MASt3R → VGGT 演进路径？
- 各自的输入输出、速度、精度？
- 哪些模型开源且可复现？

**必读**：
- [ ] Zhang et al., "Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT", arXiv 2507.08448 — **先读这篇**
- [ ] VGGT (Wang et al., CVPR 2025) — 代码：`github.com/facebookresearch/vggt`
- [ ] DUSt3R (CVPR 2024)

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
- 现有 demo 使用什么 sensor 输入？
- 3D 重建在 perception pipeline 中的角色？
- demo 的最小可用形式是什么？

**阅读/参考清单**：
- [ ] EA3D (Extract Anything 3D, arXiv 2510.25146) — 在线 3D 物体提取
- [ ] SAB3R (CVPR 2025 workshop) — "Map and Locate" task
- [ ] RAZER (arXiv 2505.15373) — open-vocabulary panoptic reconstruction
- [ ] LOCATE 3D (Meta FAIR, 2025) — 语言驱动的 3D 物体定位

---

## Baseline 选择（待调研后确认）

| 模块 | 候选 | 状态 |
|------|------|------|
| 前景分割 | SAM2 / YOLO+SAM2 | 待确定 |
| 前景重建 | 3DGS / MVSplat | 待确定 |
| 背景重建 | VGGT / DUSt3R | 待确定 |
| 融合策略 | 自行设计（参考 Unbounded-GS / HybridGS） | 待设计 |

## 前 5 篇优先阅读顺序

1. Kerbl et al., "3D Gaussian Splatting", SIGGRAPH 2023
2. Zhang et al., "Review of Feed-forward 3D Reconstruction", arXiv 2025
3. Wang et al., "VGGT", CVPR 2025
4. Zhu et al., "ObjectGS", ICCV 2025
5. Kirillov et al., "SAM 2", 2024
