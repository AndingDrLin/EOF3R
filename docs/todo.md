# 任务清单

> 按 8 阶段组织。已完成项保留勾选。

---

## Stage 0：项目初始化

- [x] 创建目录结构
- [x] 创建 .gitignore
- [x] 创建 README.md
- [x] 创建 requirements.txt（骨架）
- [x] 创建 docs/project_scope.md
- [x] 创建 docs/roadmap.md
- [x] 创建 docs/lit_review.md
- [x] 创建 docs/standards.md
- [x] 创建 docs/todo.md
- [x] 创建 docs/project_audit.md（方向诊断）
- [x] 创建 docs/experiments.md（导航实验设计）
- [x] 创建 docs/engineering.md（三阶段工程规划）
- [x] 创建 docs/risks.md（风险评估）
- [x] 创建 lit_notes/_template.md
- [x] 创建 experiments/exp_template.md
- [x] 创建 configs/default.yaml
- [x] 更新 configs/default.yaml（加 robot/cloud/costmap/safety）
- [x] 更新 CLAUDE.md（新方向 + 新目录 + BEV 约定）
- [x] 重写 README.md（新方向 + 系统架构）
- [x] 创建 src/__init__.py
- [ ] 创建 conda 环境 `eof3r` 并安装 requirements.txt
- [ ] 测试基础环境可运行（import torch, open3d 等）
- [ ] Stage 0 收尾 commit

---

## Stage 1：文献调研与基线选定

### 核心论文
- [ ] Zhang et al., "Review of Feed-forward 3D Reconstruction" (arXiv 2025) — **先读**
- [ ] Kerbl et al., "3DGS" (SIGGRAPH 2023) — 笔记
- [ ] Wang et al., "VGGT" (CVPR 2025) — 笔记
- [ ] MVSplat (ECCV 2024) — 笔记 + 代码实验
- [ ] SAM 2 论文 + 官方文档

### 各方向调研
- [ ] A 方向（3DGS 基础）笔记完成
- [ ] B 方向（Object-level 3DGS）至少 3 篇笔记
- [ ] C 方向（Feedforward 3R）至少 3 篇笔记
- [ ] D 方向（场景分割）调研完成
- [ ] E 方向（混合表示）至少 3 篇笔记
- [ ] F 方向（Autonomous Demo）调研完成
- [ ] G 方向（Feedforward/Sparse-View 3DGS）至少 3 篇笔记
- [ ] H 方向（Semantic 3DGS）至少 2 篇笔记
- [ ] I 方向（BEV Occupancy & Costmap）精读 Nav2 文档
- [ ] J 方向（ROS2 Nav2 Local Planning）精读架构文档
- [ ] K 方向（Edge-Cloud Robotics）浏览 FogROS2
- [ ] L 方向（Campus Delivery）浏览 Starship/Nuro/美团

### 决策
- [ ] 确定各模块 baseline 选择
- [ ] 更新 baselines/registry.yaml
- [ ] 更新 lit_review.md 为完整状态

---

## Stage 2：数据准备与场景分解

- [ ] 确认 $EOF3R_DATA 环境变量和数据路径
- [ ] 下载/准备 ScanNet++ 参考数据（至少 2 个场景）
- [ ] 预约 Husky + 校园场地
- [ ] 录制 campus rosbag（至少 3 个场景）
- [ ] 写 rosbag 解析脚本（`scripts/preprocess/extract_frames.py`）
- [ ] SAM2 分割在目标场景上验证
- [ ] 前景 mask 提取 pipeline
- [ ] 物体 crop 提取（多帧关联）
- [ ] 背景 mask 生成
- [ ] Gazebo/Isaac 仿真场景搭建（可选）

---

## Stage 3：前景 Object-level 3DGS 重建

- [ ] 搭建 MVSplat 推理环境
- [ ] 写 MVSplat wrapper（`src/foreground/mvsplat_wrapper.py`）
- [ ] 测试：单物体 + 2-4 crops → Gaussian .ply
- [ ] 3D 几何精度评估（Chamfer, F-Score vs GT mesh）
- [ ] 物体参数提取：3D center, size, orientation, BEV footprint
- [ ] Fallback: per-object 3DGS 优化 wrapper
- [ ] 对比：feedforward vs optimization 的精度/速度

---

## Stage 4：背景 3R 粗重建

- [ ] 搭建 VGGT 推理环境
- [ ] 写 VGGT wrapper（`src/background/vggt_wrapper.py`）
- [ ] 搭建 MASt3R/DUSt3R fallback 环境
- [ ] 输出：pointmap + 相机位姿 + 地面估计
- [ ] 地面平面估计 + 可通行区域 mask 生成
- [ ] 坐标系统一（Y-up）

---

## Stage 5：融合与 BEV 代价地图生成

- [ ] 前景-背景坐标对齐验证
- [ ] Object Gaussian → BEV 占据网格投影实现
- [ ] 语义/风险层级生成（类别→风险等级→膨胀半径）
- [ ] Costmap inflation 参数调优
- [ ] Nav2 costmap layer plugin 开发（`src/costmap/`）
- [ ] 可视化：BEV 代价地图叠加原始图像验证

---

## Stage 6：车-云异步架构

- [ ] 通信协议设计（关键帧格式、costmap patch 格式）
- [ ] 车端关键帧选择 + 上传节点
- [ ] 云端推理服务器（HTTP/gRPC）
- [ ] 云端 pipeline 集成（Stage 2-5 串联）
- [ ] 车端 costmap patch 接收 + 融合节点
- [ ] 延迟监测 + 超时丢弃逻辑
- [ ] 通信中断降级 + 重连测试

---

## Stage 7：实验验证与消融

### 实验 1：窄通道 + 不规则障碍物
- [ ] 场景搭建
- [ ] 5 次重复 × (baseline + enhanced) 运行
- [ ] 指标计算 + 可视化
- [ ] 统计检验

### 实验 2：低矮/不规则障碍物
- [ ] 场景搭建
- [ ] 测试运行 + 指标

### 实验 3：行人/自行车低速交互
- [ ] 场景搭建
- [ ] 测试运行 + 指标

### 消融
- [ ] costmap 去掉语义层
- [ ] costmap 去掉物体形状层

---

## Stage 8：论文写作与答辩

- [ ] 论文初稿
- [ ] 定量结果表
- [ ] 可视化图（≥300 DPI）
- [ ] 系统架构图
- [ ] Demo 视频
- [ ] 答辩 PPT
