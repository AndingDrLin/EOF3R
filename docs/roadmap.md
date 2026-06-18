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

## Stage 3：跨模型几何蒸馏 — Planning-Oriented 前馈高斯占据（核心创新）

> 不再"拼接"VGGT+MVSplat 做推理。改为用 VGGT 的几何输出 **训练** MVSplat 的新 decoder head。
> VGGT 是训练时的 teacher，MVSplat 是推理时的唯一模型。

### 路线

- [x] Phase A：Sequential Baseline — 三模型串行推理（用于消融对比）
  - [x] MVSplat wrapper: build/infer/extract_occupancy
  - [x] VGGT wrapper: from_pretrained + 6D 位姿解码 + 地面估计
  - [x] 坐标对齐（OpenCV→Y-up）+ scale recovery
  - [x] 消融实验（4 变体 × 3 帧配对）
- [ ] Phase B：MVSplat Decoder Retraining — 用 VGGT 几何信号训练新 head
  - [ ] 添加 occupancy head（sigmoid, 输出 0=free / 1=occupied）
  - [ ] 添加 semantic head（per-Gaussian class logits）
  - [ ] 添加 confidence head（epistemic uncertainty）
  - [ ] 损失：L_depth(VGGT) + L_occ + L_semantic(SAM2 masks) + λ·L_color
  - [ ] Freeze MVSplat encoder（cost volume），只训练 decoder head
- [ ] Phase C：可微 BEV 边缘化 + Free-Space Carving
  - [ ] 将 numpy BEV 投影替换为 torch 可微操作
  - [ ] 解析 Σ→XZ 投影（保留协方差结构）
  - [ ] VGGT 光线 free-space carving（FREE/OCCUPIED/UNKNOWN 三值）
- [ ] Phase D：端到端 Planning Loss
  - [ ] costmap 质量指标作为可微损失
  - [ ] backprop 到 Gaussian 参数

### 产出

- [x] Sequential baseline 代码（`src/foreground/mvsplat_wrapper.py`）
- [x] 消融实验脚本与结果（`ablation_study.py`, `ablation_summary.json`）
- [ ] MVSplat decoder head（occupancy + semantic + confidence）
- [ ] VGGT 几何监督数据 pipeline
- [ ] 可微 BEV 投影模块
- [ ] 论文三消融实验

---

## Stage 4：VGGT 几何监督提取（角色转变）

> VGGT 从"推理阶段"转变为"训练监督源"。
> 不再在推理时运行 VGGT，只在训练时用它提取 depth/pointmap/free-space rays。

### 路线

- [x] VGGT 推理 pipeline 搭建完成
- [x] Depth / pointmap / 相机位姿提取
- [x] 坐标系统一（OpenCV→Y-up）+ scale recovery
- [x] 地面平面估计 + 可通行区域
- [ ] **提取训练监督信号**：
  - [ ] Per-pixel depth → binary silhouette（occupancy 监督）
  - [ ] Pointmap → 3D 点密度约束（scale 正则化）
  - [ ] Free-space rays → FREE/OCCUPIED/UNKNOWN mask（三值监督）
- [ ] 构建训练数据集：Re10k 场景 → (图像, VGGT 几何监督) 对

### 产出

- [x] VGGT wrapper（`src/background/vggt_wrapper.py`）
- [ ] 训练数据预处理脚本（`scripts/preprocess/extract_vggt_supervision.py`）
- [ ] 几何监督 dataloader

---

## Stage 5：语义 + BEV + Costmap 输出

- [x] YOLO+SAM2 → 真实 COCO 语义标签（训练监督）
- [x] 动态 BEV grid + Nav2 costmap 生成（baseline 版本）
- [ ] 可微 BEV 投影（Phase C，替代 numpy 版本）
- [ ] Semantic BEV：per-Gaussian class → 投票→ BEV semantic grid
- [ ] Nav2 costmap layer plugin（ROS2 集成 — Stage 6）
- [ ] 3 消融实验 + 定量评估

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
