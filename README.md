# EOF3R

**EOF3R: Planning-Oriented Gaussian Occupancy for Low-Speed Robot Navigation**

Efficient Object-level Feedforward 3D Reconstruction with 3DGS — repurposed for robot perception.
**不追求逼真重建，追求适合规划的几何-语义表示。**

本科毕业设计原型系统。

---

## 核心思路

**跨模型几何蒸馏（Cross-Model Geometric Distillation）**：

```
训练时：  图像 → VGGT (teacher) → depth/pointmap/rays → 几何监督 ─┐
         图像 → SAM2/YOLO → 2D masks → 语义监督 ────────────────┤
                                                                  ▼
         图像 → MVSplat (student) → 学习预测 occupancy + semantic + confidence

推理时：  图像 → MVSplat → BEV occupancy + semantic costmap（单模型前馈）
```

**不是"拼接预训练模型"做串行推理，而是用几何模型（VGGT）教渲染模型（MVSplat）学会预测规划导向的占据表示。**

**训练目标**：L_depth + L_occ + L_free + L_semantic（主），L_color（辅助, λ=0.1）
**RGB/photometric 是辅助监督，不是核心目标。**

**车端**：始终独立运行本地安全回路（相机/里程计/IMU/急停/Nav2 局部规划/cmd_vel）
**云端**：运行改造后的 MVSplat（单模型前馈推理），输出 BEV occupancy + semantic costmap

---

## 核心贡献

**不是"能避障"**。本地避障（LiDAR + Nav2 obstacle layer）已经可以安全停车和基本绕行。

**技术创新**：
- **跨模型几何蒸馏**：发现 photorealistic Gaussian primitives 的 opacity 与颜色纠缠→BEV 投影不可用；提出用 VGGT 的几何信号重新训练 MVSplat decoder，解耦占据与外观
- **Planning-Oriented Gaussian 表征**：occupancy head 替代 opacity、可微 BEV 边缘化保留协方差、free-space carving 引入三值空间分类
- **训练/推理不对称架构**：VGGT 只在训练时作为几何 teacher，推理时只跑 MVSplat（单模型）

---

## 系统架构

```
┌──────────────────────────────┐       ┌──────────────────────────┐
│        Husky 车端             │       │      云端 GPU 服务器       │
│                              │       │                          │
│  相机 ─→ 关键帧选择 ─────────|─ HTTP/gRPC ─→│ MVSplat (改造后)    │
│                              │       │   occupancy prediction   │
│  本地避障 ←── Nav2 局部规划   │       │   + semantic costmap     │
│    ↑                         │       │              │           │
│    │                         │       │              │           │
│  LiDAR + odom + IMU          │       │              ↓           │
│    +                          │       │                          │
│  云端 costmap (可选增强) ←───|────────|── costmap patch (异步)    │
│                              │       │                          │
│  急停 ← 安全控制器 (独立)      │       │  (结果延迟 >3s → 丢弃)    │
│  cmd_vel → 电机 (≤0.5 m/s)   │       │  (通信中断 → 降级本地避障) │
└──────────────────────────────┘       └──────────────────────────┘
```

**推理时只跑一个模型（MVSplat）**。VGGT 和 SAM2/YOLO 只在训练时提供几何/语义监督。

---

## 当前状态

- [x] Stage 0：项目初始化（目录结构、文档骨架、配置、工具链）
- [x] 方向调整：从纯 3D 重建扩展为机器人感知系统（2026-05）
- [x] Stage 1：文献调研完成（12 个方向，24 篇笔记）
- [ ] Stage 2：数据准备（Re10k 公开数据可用，待录制 campus rosbag）
- [x] **Phase A：Sequential Baseline** ✅ — E2E 跑通，消融完成。IoU=0.052, cov=1.88%, lethal=55%
- [ ] **Phase B：MVSplat Decoder Retraining** 🔜 — 跨模型几何蒸馏，VGGT 作 teacher（下一步）
- [ ] Phase C：可微 BEV + Free-Space Carving
- [ ] Phase D：端到端 Planning Loss
- [x] 统一环境：`eof3r` conda env (Python 3.10, torch 2.5.1, CUDA 12.1, RTX A6000)
- [ ] Stage 6：车-云异步架构
- [ ] Stage 7：实验验证与消融
- [ ] Stage 8：论文写作与答辩

---

## 快速开始

```bash
# 1. 创建环境
conda create -n eof3r python=3.10 -y && conda activate eof3r
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 安装模型依赖（三个模型均为开源，pip 一键安装）
pip install git+https://github.com/facebookresearch/sam2.git      # SAM2 分割
pip install git+https://github.com/facebookresearch/vggt.git      # VGGT 背景几何
# MVSplat 需 clone 到本地后 export MVSPLAT_ROOT=/path/to/mvsplat
# https://github.com/donydchen/mvsplat

# 3. 安装本项目的依赖
pip install -r eof3r/requirements.txt

# 4. 运行 E2E 测试
python eof3r/scripts/eval/test_e2e_pipeline.py
```

## 环境

- **开发环境**: `conda activate eof3r` — Python 3.10, PyTorch 2.5.1, CUDA 12.1, RTX A6000 (48GB)
- **无 GPU 测试**: 使用 `--skip-mvsplat` 标志跳过 MVSplat，使用合成高斯球
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
