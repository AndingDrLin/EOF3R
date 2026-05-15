# 项目诊断报告

> 生成日期：2026-05-15
> 目的：记录项目方向从"纯混合 3D 重建"调整为"低速校园配送机器人感知系统"的决策过程，诊断当前状态与最新方向之间的差距。

---

## 1. 方向变更记录

### 原始方向（Stage 0 初始化时）

纯计算机视觉研究项目：前景物体用 3DGS 精细重建 + 背景用 feedforward 模型（VGGT/DUSt3R）快速重建 → 融合 → Open3D 可视化 demo。核心评估指标是 PSNR/SSIM/Chamfer。

### 新方向（2026-05 确定）

面向 Husky 低速无人车的园区配送感知系统。技术核心不变（前景 object-level 3DGS + 背景 3R 重建 + 融合），但输出目标从"渲染图像"变为"BEV 语义占据代价地图"，应用场景从"3D 可视化"变为"局部路径规划增强"，系统从"单机 Python pipeline"变为"车端-云端异步协作"。

### 变更原因

1. 纯 3D 重建方向与现有大量论文重叠度高，差异化不足
2. 导师/实验室有 Husky 无人车平台，可以支撑机器人相关实验
3. 低速近场配送是真实需求场景，几何-语义增强对规划质量的影响是有意义的研究问题
4. feedforward 3DGS 的速度优势在机器人场景中更有价值

---

## 2. 当前项目状态

### 2.1 目录结构

```
EOF3R/
├── README.md              # 简短，描述旧方向
├── CLAUDE.md              # 400行，12节，质量高，§3-§12 可直接保留
├── requirements.txt       # ML/3D 包，无 ROS2
├── pyproject.toml         # ruff 配置，无问题
├── .pre-commit-config.yaml
├── .gitignore             # 合理，缺 ROS2 build 产物
│
├── docs/                  # 5 个文档，写了旧方向
│   ├── project_scope.md   # 旧范围
│   ├── roadmap.md         # 7 阶段，Stage 6 = Open3D 可视化
│   ├── lit_review.md      # 6 方向，缺 BEV/costmap/Nav2/云端
│   ├── standards.md       # 通用规范，可直接保留
│   └── todo.md            # 旧任务清单
│
├── src/                   # 只有 __init__.py（一行注释）
│   ├── segmentation/      # 空目录
│   ├── foreground/        # 空目录
│   ├── background/        # 空目录
│   ├── fusion/            # 空目录
│   ├── demo/              # 空目录
│   └── utils/             # 空目录
│
├── configs/
│   ├── default.yaml       # 完整 pipeline 配置，缺 robot/cloud/costmap
│   ├── plot_style.yaml    # 论文绘图风格，可保留
│   ├── data/              # 空
│   └── model/             # 空
│
├── baselines/
│   ├── registry.yaml      # 4 条基线（全注释）
│   └── patches/           # 空
│
├── scripts/               # 空（仅 preprocess/ 和 eval/ 子目录）
├── tests/                 # 空
├── data/                  # 空（仅 raw/ 和 processed/ 子目录）
├── outputs/               # 空
├── lit_notes/             # 仅 _template.md
├── experiments/           # 仅 exp_template.md
└── thesis/                # 5 个空子目录
```

### 2.2 代码状态

- **Python 代码行数**：0（仅 `src/__init__.py` 有一行注释）
- **脚本**：0 个
- **测试**：0 个
- **基线克隆**：0 个
- **数据**：0
- **环境**：尚未创建 conda 环境

### 2.3 Git 状态

- 分支：`main`（与 `origin/main` 同步）
- 提交：2 次
  - `ed41405` — init: project structure, constitution, and tooling setup
  - `de0baa0` — fix: track CLAUDE.md, exclude only .claude/ from version control
- 工作区：干净

---

## 3. 一致性矩阵

| 模块/概念 | 旧方向 | 新方向 | 一致性 | 处理 |
|-----------|--------|--------|--------|------|
| 前景 3DGS 重建 | per-object 优化训练 | feedforward 优先，优化 fallback | 部分一致 | 调整 |
| 背景重建 | VGGT / DUSt3R | VGGT / DUSt3R / MASt3R | 一致 | 扩展 |
| 前景分割 | SAM2 | SAM2 | 一致 | 保留 |
| 融合 | rendering 融合 | BEV 投影 + costmap 融合 | 不一致 | 重写 |
| Demo | Open3D 可视化 | Husky + ROS2 RViz 导航 | 不一致 | 重写 |
| 评估指标 | PSNR, SSIM, Chamfer | 上述 + path_length, stop_count 等 | 部分一致 | 扩展 |
| 数据集 | ScanNet++, Replica | 上述 + campus rosbag, Gazebo | 部分一致 | 扩展 |
| 系统架构 | 单机 Python | 车端-云端异步 | 不一致 | 新增 |
| 3D 约定 | Y-up, OpenGL | Y-up(重建) + Z-up(BEV) | 部分一致 | 加 BEV 约定 |
| 编码规范 | ruff, 类型标注, snake_case | 不变 | 一致 | 保留 |

---

## 4. 保留清单

以下内容与方向无关，可直接保留：

| 文件/目录 | 理由 |
|-----------|------|
| CLAUDE.md §3-§12 | 基线管理、数据管理、实验管理、代码标准、3D 约定、论文材料、Git 工作流、环境、可复现性、性能原则 — 全部领域无关 |
| docs/standards.md | 通用工程规范 |
| configs/plot_style.yaml | 论文绘图风格（需加导航指标标签） |
| experiments/exp_template.md | 实验日志模板（需加机器人字段） |
| lit_notes/_template.md | 论文笔记模板 |
| pyproject.toml | 项目元数据和 ruff 配置 |
| .pre-commit-config.yaml | 预提交钩子 |
| src/__init__.py | 包声明 |
| src/ 下所有空子目录 | 模块边界合理，只需加 costmap/ 和 communication/ |

## 5. 重构清单

| 文件 | 重构程度 | 说明 |
|------|----------|------|
| README.md | 重写 | 新方向、系统架构图、文档索引 |
| docs/project_scope.md | 重写 | 新范围、安全声明、导航实验 |
| docs/roadmap.md | 重写 | 7→8 阶段，内容大变 |
| docs/lit_review.md | 大改 | 加 6 个新方向 |
| docs/todo.md | 重写 | 按新路线重新列任务 |
| configs/default.yaml | 大改 | 加 robot/cloud/costmap/safety 段 |
| CLAUDE.md §1-§2 | 小改 | 更新项目描述、加新目录 |
| baselines/registry.yaml | 更新 | 加 feedforward 3DGS、MASt3R 等 |
| .gitignore | 小改 | 加 ROS2 build 产物 |

## 6. 新增清单

| 新增 | 类型 | 理由 |
|------|------|------|
| docs/project_audit.md | 文档 | 本文件 — 诊断记录 |
| docs/experiments.md | 文档 | 导航实验设计 |
| docs/engineering.md | 文档 | 三阶段工程规划 |
| docs/risks.md | 文档 | 风险评估与降级方案 |
| src/costmap/ | 代码 | BEV 投影、占据网格、costmap 生成 |
| src/communication/ | 代码 | 车端-云端异步通信 |
| configs/robot/ | 配置 | Nav2 参数、传感器配置 |
| configs/cloud/ | 配置 | 云端推理服务配置 |
| scripts/robot/ | 脚本 | Husky launch 文件 |
| data/README.md | 文档 | 数据格式说明 |
| outputs/README.md | 文档 | 输出目录约定 |

## 7. 缺失模块

按优先级排列：

1. **文档更新**（最优先 — 指导后续工作）
2. **configs/default.yaml 更新**（定义模块接口）
3. **src/segmentation/** — SAM2 wrapper
4. **src/foreground/** — 3DGS/MVSplat wrapper
5. **src/background/** — VGGT/DUSt3R/MASt3R wrapper
6. **src/fusion/** — 坐标对齐 + BEV 投影
7. **src/costmap/** — 占据网格、语义层级、costmap 输出
8. **src/communication/** — 异步通信、超时、降级
9. **src/demo/** — ROS2 Nav2 节点
10. **实验脚本** — rosbag 录制、离线评估、可视化

## 8. 关键设计决策

### 为什么不做完整自动驾驶

完整自动驾驶需要处理高速(>30 km/h)、开放道路、动态交通流、交通规则、路口决策等。本项目只针对校园/园区内 <0.5 m/s 的低速近场配送，场景范围小、速度低、安全边界清晰。

### 为什么云端不直接控制车辆

车辆安全不能依赖不可靠的网络。云端结果必然有延迟（秒级），而避障需要毫秒级响应。因此本地安全回路（急停、Nav2 局部规划、cmd_vel）必须独立运行。云端结果只能作为 costmap 增强，改善局部规划的质量，不能替代本地规划或直接输出控制指令。

### 为什么用 feedforward 3DGS 而不是优化式 3DGS

传统 3DGS 需要每个场景迭代训练（7000+ 次迭代，分钟级），不适合无人车"看几秒钟就要用"的场景。Feedforward 3DGS（如 MVSplat）可以在单次前向传播中预测 3DGS 参数，速度从分钟级降到秒级。如果 feedforward 质量不够，回退到 per-object 优化作为下限。

### 为什么核心贡献是"路径质量"而不是"避障能力"

本地避障（LiDAR + 超声波 + Nav2 obstacle layer）已经可以让车安全停下或绕开障碍物。但纯几何避障会把所有未知物体当作刚性墙壁，导致：
- 过度保守（离障碍物很远就停下）
- 不合理绕行（不知道物体实际形状，绕大圈）
- 路径抖动（连续帧的几何噪声导致 costmap 不稳定）

本项目的贡献是：用 object-level 3DGS 提供更准确的物体占据形状，用语义提供风险分级，从而让局部路径规划做出更合理、更平滑、更高效的决策。

---

## 9. 推荐下一步行动

### 立即（本次）

1. 重写 `docs/project_scope.md`
2. 创建 `docs/risks.md`
3. 重写 `docs/roadmap.md`

### 后续

4. 更新 `docs/lit_review.md`（加 6 个新方向）
5. 创建 `docs/experiments.md`
6. 创建 `docs/engineering.md`
7. 更新 `configs/default.yaml`
8. 更新 `CLAUDE.md` §1-§2
9. 重写 `README.md`
10. 重写 `docs/todo.md`
11. 创建新目录和 `__init__.py` 文件
12. 更新 `baselines/registry.yaml`、`.gitignore` 等

---

## 10. 风险提示

- **范围扩大风险**：加入了机器人、云端、ROS2 等新组件，系统复杂度显著增加。需要用 risks.md 和 engineering.md 明确降级路径。
- **文档一致性风险**：5+ 个新文档需要保持互不矛盾。每次修改文档后应检查关键约束是否一致。
- **过早实现风险**：文档和配置未稳定之前不应开始写代码。Phase 1（离线验证）之前有充分的文档规划时间。
