# 技术路线

> 8 阶段路线，总计约 16 周。Stage 0 已完成。Stage 3-4 可并行。

---

## Stage 0：项目初始化（第 1 周）✅ 已完成

- [x] 目录结构与项目骨架
- [x] CLAUDE.md（项目宪法）
- [x] docs/（project_scope, roadmap, lit_review, standards, todo）
- [x] configs/default.yaml、plot_style.yaml
- [x] baselines/registry.yaml
- [x] pyproject.toml（ruff 配置）
- [x] .pre-commit-config.yaml
- [x] .gitignore
- [x] 创建 conda 环境 `eof3r` 并安装依赖
- [x] 验证环境可运行（SAM2 + VGGT + MVSplat 全部通过 E2E）

---

## Stage 1：文献调研与基线选定（第 2-5 周）

### 调研方向（12 个）

- A. 3DGS 基础（3DGS, Mip-Splatting, Scaffold-GS, 2DGS）
- B. Object-level 3DGS（ObjectGS, GES, Gaussian Grouping）
- C. Feedforward 3D 重建（VGGT, DUSt3R, MASt3R, Spann3R）
- D. 场景分割（SAM, SAM2, YOLO-world）
- E. 混合 3D 场景表示（Unbounded-GS, HybridGS）
- F. Autonomous Perception Demo（参考实现）
- **G. Feedforward / Sparse-View 3DGS**（MVSplat, pixelSplat, latentSplat, Flash3DGS）
- **H. Semantic 3DGS**（LangSplat, LEGaussians, Feature-3DGS, OpenGaussian）
- **I. BEV Occupancy & Costmap**（BEVDet, Occ3D, Nav2 costmap_2d）
- **J. ROS2 Nav2 Local Planning**（DWB, TEB, RPP, costmap plugins）
- **K. Edge-Cloud Robotics**（FogROS2, Rapyuta, cloud latency handling）
- **L. Campus/Last-Mile Delivery Perception**（Starship, Nuro, 美团无人配送）

### 产出

- [x] 更新 `docs/lit_review.md`（完整论文索引，12 方向 24 篇笔记）
- [x] `lit_notes/` 下 24 篇核心论文的阅读笔记
- [x] 确定每个模块的 baseline 选择（优先 feedforward 方案）
- [ ] 更新 `baselines/registry.yaml`（MVSplat/DepthSplat/SAM2/VGGT 已登记，待补 DUSt3R/MASt3R）

---

## Stage 2：数据准备与场景分解（第 5-7 周）

### 数据

- **参考数据**：ScanNet++（室内，有 pose，用于重建精度验证）
- **目标数据**：校园 Husky rosbag（自采集，至少 3 个场景）
- **仿真数据**（可选）：Gazebo / Isaac Sim 合成场景

### 场景分解

- [ ] 确认 SAM2 在目标场景上的分割质量
- [ ] 前景 mask 提取（box prompt 或 automatic mode）
- [ ] 背景 mask 生成（inpaint 前景区域）
- [ ] 物体 crop 提取（从多帧中裁剪前景物体区域）
- [ ] rosbag 格式解析与关键帧提取脚本
- [ ] 数据预处理 pipeline（`scripts/preprocess/`）

### 产出

- [ ] 至少 3 个场景的预处理数据（前景 masks + 物体 crops + 背景 masks + 相机信息）
- [ ] 数据处理脚本

---

## Stage 3：前景 G2O-inspired Feedforward Gaussian Occupancy（第 7-10 周）

> 与 Stage 4 可部分并行（第 8-10 周重叠）
> 核心方法：融合 G2O 几何约束思想的前馈式 object-level Gaussian occupancy 预测。
> 前景与背景分开表示；前景负责物体级占据、语义、风险和 footprint，背景由 Stage 4 负责粗几何、free-space、unknown-space 与 occlusion boundary。
> 不使用逐场景 30k 迭代优化，不使用高阶 SH/view-dependent color 作为核心输出。

### 优先路线：G2O-inspired Feedforward Gaussian Occupancy

- [x] 搭建 MVSplat 前馈推理 pipeline（wrapper: build/infer/extract_occupancy）
- [ ] 引入 G2O 思想：geometry scaffold 约束前馈 decoder、opacity/confidence-aware 高斯筛选（训练阶段，待 implement）
- [x] 输入：RGB 多帧图像（2-4 视角）— 当前用 Re10k 全图验证
- [x] 输出：Gaussian primitives（means, opacities, scales, covariances）— 坐标系统待校准
- [ ] 核心损失：L_occupancy + L_mask + L_depth + L_silhouette + L_footprint + L_semantic + L_confidence（训练阶段）
- [ ] 辅助损失：L_rgb（权重 0.1，仅用于诊断）
- [ ] 评估几何/占据精度（Chamfer, F-Score, Footprint IoU）— 坐标校准后进行

### Fallback 路线：Per-Object 优化 3DGS + 后处理提取 Occupancy

- [ ] 如果 feedforward 质量不够，回退到每个物体独立训练 3DGS
- [ ] 从优化后的 3DGS 中后处理提取 occupancy_alpha、BEV footprint
- [ ] 仍比全场景训练快（单个物体 + 少量迭代）

### 产出

- [x] 前景 Gaussian occupancy 模块（`src/foreground/mvsplat_wrapper.py`）
- [x] Gaussian 输出（GaussianData dataclass 含 means, opacities, scales, covariances）
- [x] E2E 测试脚本（`scripts/eval/test_e2e_pipeline.py`，38 个定量指标）

---

## Stage 4：背景 3R / VGGT-like 前馈世界表征估计（第 8-10 周）

> 与 Stage 3 可部分并行。目标是从 RGB、LiDAR / depth、odometry 等多模态输入中端到端预测背景状态描述（pointmap + 地面 + free-space + unknown-space + occlusion boundary + 可通行区域），不是逼真背景渲染。

### 路线

- [x] 搭建 VGGT 推理 pipeline（wrapper: VGGT.from_pretrained, 6D 位姿解码, 地面估计）
- [x] 输入：场景全图 — Re10k 4 帧 720p 图像
- [x] 输出：dense pointmap + 相机位姿 + 地面平面估计 + 可通行区域 mask
- [ ] 坐标系统一到项目约定（右手 Y-up）— 🔴 当前核心问题（BEV coverage 0.45%）
- [ ] 可选：将 3R pointmap 作为 Stage 3 前馈 decoder 的 geometry hint
- [ ] Fallback: DUSt3R/MASt3R（待搭建）

### 产出

- [x] 背景几何估计模块（`src/background/vggt_wrapper.py` + `vggt_stub.py`）
- [x] 背景 pointmap + 地面平面 + 相机 pose

---

## Stage 5：融合与 BEV 代价地图生成（第 10-12 周）

### 路线

- [ ] 前景-背景坐标对齐 — 🔴 当前核心问题，MVSplat/VGGT 坐标系不一致
- [x] Gaussian occupancy → BEV 占据网格投影算法（矢量化 bincount + gaussian_filter, 650x 加速）
- [x] 融合 object-level 前景表示与背景状态表示（架构就绪，坐标校准后验证）
- [x] 统一输出：BEV occupancy、semantic costmap（架构就绪）
- [x] 语义/风险层级生成（类别 → 风险等级 → 膨胀半径，semantic_weights 已定义）
- [x] Nav2 兼容的 uint8 costmap 格式输出（0=free, 254=lethal, 255=unknown）
- [ ] Nav2 costmap layer plugin（ROS2 节点适配 — 待 Stage 6）
- [x] 可视化：E2E pipeline 可视化（`e2e_pipeline_visualization.png`）

### 产出

- [x] 融合模块（`src/fusion/bev_projector.py` + `coord_utils.py`）
- [x] Costmap 生成模块（`src/costmap/costmap_generator.py`）
- [ ] Nav2 costmap layer plugin（待 ROS2 集成）

---

## Stage 6：车-云异步架构（第 11-13 周）

> 与 Stage 5 可部分并行

### 车端

- [ ] ROS2 本地安全节点（急停、速度限制、心跳）
- [ ] 关键帧选择与上传节点（异步，非阻塞）
- [ ] Costmap patch 接收与融合节点
- [ ] Latency 监测与超时丢弃逻辑

### 云端

- [ ] 推理服务器（HTTP 或 gRPC）
- [ ] 异步任务队列（接收关键帧 → 调用 Stage 2-5 pipeline → 返回 costmap patch）
- [ ] 超时和错误处理

### 通信

- [ ] 通信协议定义（关键帧格式、costmap patch 格式）
- [ ] 降级逻辑：延迟 >3s 丢弃、通信中断降级为本地避障、恢复后自动重连

### 产出

- [ ] 通信模块（`src/communication/`）
- [ ] Demo 模块更新（`src/demo/` → ROS2 节点）
- [ ] 系统降级测试

---

## Stage 7：实验验证与消融（第 13-15 周）

### 三个导航实验场景

1. **窄通道 + 不规则障碍物**（自行车斜放、电动车侧倒、纸箱堆叠）
2. **低矮/不规则障碍物**（倒地自行车、路沿、低障碍栏、压扁纸箱）
3. **行人/自行车低速交互**（行人慢速横穿、自行车低速经过）

### 实验设计

- [ ] 每个场景 5 次重复，交替运行 baseline 和 enhanced
- [ ] Baseline：Nav2 local planner + 纯 LiDAR obstacle layer
- [ ] Enhanced：Nav2 local planner + LiDAR + 云端 costmap 增强层
- [ ] 消融：costmap 去掉语义层、costmap 去掉物体形状层
- [ ] 消融：纯 photometric loss (L_rgb only) vs planning-oriented loss (无 occupancy/mask/silhouette loss)

### 评估指标

- **安全性**：collision / near-collision count
- **效率**：time to goal, path length
- **质量**：path smoothness (integrated curvature), unnecessary stop count, minimum clearance
- **感知精度**：footprint IoU, costmap IoU, occupancy accuracy, 3D bbox IoU, boundary IoU
- **系统**：latency (mean, p95, max), cloud timeout rate
- **辅助**：PSNR/SSIM/LPIPS（diagnostic only，不是项目核心指标）

### 产出

- [ ] 实验日志（`experiments/`）
- [ ] 定量结果表 + 路径可视化对比图
- [ ] 消融分析

---

## Stage 8：论文写作与答辩（第 14-16 周）

- [ ] 论文初稿（中文）
- [ ] 定量结果表与可视化图（≥300 DPI）
- [ ] 系统架构图
- [ ] Demo 视频录制
- [ ] 答辩 PPT

---

## 时间线总览

```
Week:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
Stage 0: ██
Stage 1:    ████████████
Stage 2:                ████████████
Stage 3:                      ████████████████
Stage 4:                      ████████████████
Stage 5:                                  ████████████
Stage 6:                                    ████████████
Stage 7:                                              ████████████
Stage 8:                                                ████████████
```

- Stage 3 和 Stage 4 可并行（都用 Stage 2 的输出，互不依赖）
- Stage 5 依赖 Stage 3+4
- Stage 6 可与 Stage 5 并行（通信模块可以用 mock 数据先开发）
- Stage 7 依赖 Stage 5+6 的模块就绪
- Stage 8 与 Stage 7 有重叠（实验跑起来就可以开始写论文）

---

## 与旧路线的主要变化

| 旧 | 新 | 变化原因 |
|----|-----|----------|
| 7 阶段 | 8 阶段 | 加了车-云架构阶段 |
| 目标：photorealistic 3D 重建 | 目标：planning-oriented Gaussian occupancy | 项目定位调整 |
| 主指标：PSNR/SSIM/LPIPS | 主指标：footprint IoU / occupancy accuracy / path quality | 评价体系调整 |
| RGB photometric loss 为主 | L_occupancy + L_mask + L_depth + L_silhouette 为主，RGB 为辅 | 训练目标调整 |
| 前馈 3DGS 为"快速精细重建" | G2O-inspired feedforward Gaussian occupancy 预测 | 方法定位调整 |
| SH/color/view-dependent 为核心输出 | 几何/语义/占据为核心输出，SH 可丢弃 | 输出定义调整 |
| Stage 6: Open3D 可视化 demo | Stage 6: 车-云异步架构 | 真实机器人场景 |
| 数据只有 ScanNet++/Replica | 加 campus rosbag + Gazebo | 需要机器人场景 |
| 融出是 rendering | 融合输出是 BEV costmap | 目标变了 |
