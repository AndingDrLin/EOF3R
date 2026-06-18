# EOF3R

**EOF3R: Planning-Oriented Gaussian Occupancy for Low-Speed Robot Navigation**

Efficient Object-level Feedforward 3D Reconstruction with 3DGS — repurposed for robot perception.
**不追求逼真重建，追求适合规划的几何-语义表示。**

本科毕业设计原型系统。

---

## 核心思路

相机输入 → SAM2 前景分割 → **前景物体** G2O-inspired feedforward Gaussian occupancy 预测（object state, 3D bbox, occupancy_alpha, BEV footprint, semantic class, risk score, confidence）
　＋ **背景** 3R 模型粗几何估计（pointmap、相机位姿、地面结构、可通行区域）
→ 融合 → **BEV semantic costmap** → ROS2 Nav2 局部路径规划增强

**训练目标**：L_occupancy + L_mask + L_depth + L_silhouette + L_footprint + L_semantic + L_confidence（主），L_rgb（辅助）
**RGB/photometric 是辅助监督，不是核心目标。SH/color/view-dependent effects 不是核心输出。**

**车端**：始终独立运行本地安全回路（相机/里程计/IMU/急停/Nav2 局部规划/cmd_vel）
**云端**：异步高算力推理（SAM2 mask 细化 / 3R 背景几何估计 / G2O-inspired feedforward Gaussian occupancy / 语义 costmap 生成）。云端返回 lightweight planning-oriented representation，不返回完整 Gaussian 渲染模型。

---

## 核心贡献

**不是"能避障"**。本地避障（LiDAR + Nav2 obstacle layer）已经可以安全停车和基本绕行。

本项目要解决的问题是：
- **更准确的物体占据形状** → 减少过度保守的绕行和停车
- **更丰富的语义风险信息** → 区分"可以靠近的路锥"和"必须远离的行人"
- **更平滑的局部路径** → 减少路径抖动和无效停顿

---

## 系统架构

```
┌──────────────────────────────┐       ┌──────────────────────────┐
│        Husky 车端             │       │      云端 GPU 服务器       │
│                              │       │                          │
│  相机 ─→ 关键帧选择 ─────────|─ HTTP/gRPC ─→│ SAM2 分割细化        │
│                              │       │       3R 背景几何估计     │
│  本地避障 ←── Nav2 局部规划   │       │       G2O-inspired       │
│    ↑                         │       │       Gaussian occupancy  │
│    │                         │       │       语义 costmap 生成   │
│    │                         │       │              │           │
│  LiDAR + odom + IMU          │       │              │           │
│    +                          │       │              ↓           │
│  云端 costmap (可选增强) ←───|────────|── costmap patch (异步)    │
│                              │       │                          │
│  急停 ← 安全控制器 (独立)      │       │  (结果延迟 >3s → 丢弃)    │
│  cmd_vel → 电机 (≤0.5 m/s)   │       │  (通信中断 → 降级本地避障) │
└──────────────────────────────┘       └──────────────────────────┘
```

---

## 当前状态

- [x] Stage 0：项目初始化（目录结构、文档骨架、配置、工具链）
- [x] 方向调整：从纯 3D 重建扩展为机器人感知系统（2026-05）
- [x] Stage 1：文献调研完成（12 个方向，23 篇笔记）
- [ ] Stage 2：数据准备（Re10k 公开数据可用，待录制 campus rosbag）
- [x] Stage 3-5：核心 pipeline (SAM2 + VGGT + MVSplat + fusion + costmap) E2E 跑通
- [x] 统一环境：`eof3r` conda env (Python 3.10, torch 2.5.1, CUDA 12.1, RTX A6000)
- [ ] Stage 6：车-云异步架构
- [ ] Stage 7：实验验证与消融
- [ ] Stage 8：论文写作与答辩

---

## 环境

- **开发环境**: `conda activate eof3r` — Python 3.10, PyTorch 2.5.1, CUDA 12.1, RTX A6000 (48GB)
- **基线环境**: `conda activate mvsplat` — PyTorch 2.1.2, CUDA 11.8 (仅 MVSplat 独立测试)
- **车端**: ROS2 Humble（Husky 车载计算机）
- 详见 `eof3r/requirements.txt`

---

## 项目结构

```
EOF3R/                              # 仓库根目录
├── README.md
├── CLAUDE.md                       # 项目宪法与开发指引
├── .gitignore
│
├── eof3r/                          # 所有可运行代码
│   ├── src/                        # 核心源码（8 个模块）
│   │   ├── segmentation/           # SAM2 分割
│   │   ├── foreground/             # MVSplat 前馈高斯占据
│   │   ├── background/             # VGGT 背景几何估计
│   │   ├── fusion/                 # BEV 投影与融合
│   │   ├── costmap/                # Nav2 代价地图生成
│   │   └── communication/          # 车-云通信（预留）
│   ├── scripts/                    # 独立脚本（eval/, setup/, robot/）
│   ├── configs/                    # YAML 配置
│   ├── tests/                      # 冒烟测试
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .pre-commit-config.yaml
│
├── docs/                           # 项目文档（9+ .md + 23 篇论文笔记）
│   ├── project_scope.md
│   ├── roadmap.md
│   ├── lit_review.md
│   ├── todo.md
│   ├── current_issues.md
│   └── lit_notes/
│
├── baselines/                      # 外部基线代码（gitignored）
├── data/                           # 数据集（gitignored）
└── outputs/                        # 实验输出（gitignored）
```

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [project_scope.md](docs/project_scope.md) | 做什么、不做什么、成功标准、安全约束 |
| [project_audit.md](docs/project_audit.md) | 方向变更诊断、一致性分析 |
| [roadmap.md](docs/roadmap.md) | 8 阶段技术路线与时间线 |
| [lit_review.md](docs/lit_review.md) | 12 方向文献调研与基线选择 |
| [experiments.md](docs/experiments.md) | 3 个导航实验场景设计 |
| [engineering.md](docs/engineering.md) | Phase 1-3 工程实现规划 |
| [risks.md](docs/risks.md) | 9 类风险与降级方案 |
| [standards.md](docs/standards.md) | 命名与工程规范补充 |
| [todo.md](docs/todo.md) | 按阶段的任务清单 |
| [current_issues.md](docs/current_issues.md) | 当前问题与根因分析 |
