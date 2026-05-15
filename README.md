# EOF3R

**Efficient Object-level Feedforward 3D Reconstruction with 3DGS**

Low-Speed Campus Delivery Robot Perception via Object-Level 3DGS and 3R Background Reconstruction.

本科毕业设计原型系统。

---

## 核心思路

相机输入 → SAM2 前景分割 → **前景物体** Feedforward 3DGS 快速精细重建（物体级 Gaussians、3D 中心、尺寸、朝向、BEV footprint）
　＋ **背景** 3R 模型粗重建（pointmap、相机位姿、地面结构、可通行区域）
→ 融合 → **BEV 语义占据代价地图** → ROS2 Nav2 局部路径规划增强

**车端**：始终独立运行本地安全回路（相机/里程计/IMU/急停/Nav2 局部规划/cmd_vel）
**云端**：异步高算力推理（SAM2 mask 细化 / 3R 背景重建 / FF 3DGS 物体重建 / 语义 costmap 生成）

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
│                              │       │       3R 背景重建         │
│  本地避障 ←── Nav2 局部规划   │       │       FF 3DGS 物体重建    │
│    ↑                         │       │       语义 costmap 生成   │
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
- [ ] Stage 1：文献调研（12 个方向）
- [ ] Stage 2：数据准备
- [ ] Stage 3-8：待开始

---

## 环境

- Python 3.10+
- CUDA 12.x
- PyTorch ≥2.0
- ROS2 Humble（车端）
- 详见 `requirements.txt`

---

## 项目结构

```
EOF3R/
├── docs/               # 项目文档（9 个 .md）
│   ├── project_scope.md    # 范围与目标
│   ├── project_audit.md    # 方向诊断
│   ├── roadmap.md          # 8 阶段技术路线
│   ├── lit_review.md       # 12 方向文献调研
│   ├── experiments.md      # 导航实验设计
│   ├── engineering.md      # 三阶段工程规划
│   ├── risks.md            # 风险评估
│   ├── standards.md        # 补充规范
│   └── todo.md             # 任务清单
├── configs/            # YAML 配置
├── src/                # 核心源码（8 个模块）
├── scripts/            # 独立脚本
├── baselines/          # 外部基线（gitignored）
├── data/               # 数据集（gitignored）
├── outputs/            # 输出（gitignored）
├── experiments/        # 实验日志
├── lit_notes/          # 论文笔记
└── tests/              # 冒烟测试
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
