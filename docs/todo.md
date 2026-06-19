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
- [x] 创建 conda 环境 `eof3r`（Python 3.10 + torch 2.5.1 + CUDA 12.1）
- [x] 测试基础环境可运行（SAM2 + VGGT + torch + numpy + scipy 全部可用）
- [x] Stage 0 收尾 commit

---

## Stage 1：文献调研与基线选定 ✅ 已完成 (2025-06-16)

### 核心论文
- [x] Zhang et al., "Review of Feed-forward 3D Reconstruction" (arXiv 2025) — 通过 Web 调研了解框架
- [x] Kerbl et al., "3DGS" (SIGGRAPH 2023) — 笔记
- [x] Wang et al., "VGGT" (CVPR 2025) — 笔记
- [x] MVSplat (ECCV 2024) — 笔记 + 代码实验（待 Stage 3）
- [x] SAM 2 论文 + 官方文档

### 各方向调研
- [x] A 方向（3DGS 基础）笔记完成：3 篇（3DGS, Mip-Splatting, 2DGS）
- [x] B 方向（Object-level 3DGS）：1 篇笔记（ObjectGS）+ 3 篇待读清单
- [x] C 方向（Feedforward 3R）：3 篇笔记（VGGT, DUSt3R, MASt3R/MUSt3R）
- [x] D 方向（场景分割）调研完成：2 篇笔记（SAM2, YOLO-World）
- [x] E 方向（混合表示）：2 篇笔记（Unbounded-GS, HybridGS）
- [x] F 方向（Autonomous Demo）调研完成：1 篇笔记（GaussianFormer）+ 2 篇待读
- [x] G 方向（Feedforward/Sparse-View 3DGS）：3 篇笔记（MVSplat, pixelSplat, latentSplat）
- [x] H 方向（Semantic 3DGS）：3 篇笔记（LangSplat, LEGaussians, Feature 3DGS）
- [x] I 方向（BEV Occupancy & Costmap）：2 篇笔记（Occ3D, Nav2 costmap_2d docs）
- [x] J 方向（ROS2 Nav2 Local Planning）：1 篇笔记（Nav2 Planning docs）
- [x] K 方向（Edge-Cloud Robotics）：1 篇笔记（FogROS2）
- [x] L 方向（Campus Delivery）：1 篇笔记（Starship）+ Nuro/美团/京东 待深入

### 决策
- [x] 确定各模块 baseline 选择
- [ ] 更新 baselines/registry.yaml（待 Stage 2/3 代码实验后最终确认）
- [x] 更新 lit_review.md 为完整状态

---

## Stage 2：数据准备与场景分解

- [ ] 确认 $EOF3R_DATA 环境变量和数据路径
- [x] 公开数据可用：Re10k 4 帧 720p 样本保存到 `data/public/re10k_samples/`
- [ ] 下载/准备 ScanNet++ 参考数据（至少 2 个场景）
- [ ] 预约 Husky + 校园场地
- [ ] 录制 campus rosbag（至少 3 个场景）
- [ ] 写 rosbag 解析脚本（`eof3r/scripts/preprocess/extract_frames.py`）
- [x] SAM2 分割在 Re10k 上验证（65 object 过分割，需调参或加 YOLO 预处理）
- [ ] 前景 mask 提取 pipeline
- [ ] 物体 crop 提取（多帧关联）
- [ ] 背景 mask 生成
- [ ] Gazebo/Isaac 仿真场景搭建（可选）

---

## Stage 3-5：跨模型几何蒸馏 — MVSplat Decoder 改造（核心创新）

> 定位转变：不再串行拼接 VGGT+MVSplat 做推理。
> VGGT = 训练时的几何 teacher，MVSplat = 推理时的唯一模型（单模型前馈）。

### Phase A：Sequential Baseline ✅ 已完成 (2026-06-19 复验)
- [x] MVSplat wrapper（build/infer/extract_occupancy, 131K Gaussians, α_mean=0.28）
- [x] VGGT wrapper（from_pretrained + 9D 位姿解码 + 地面估计, scale ×7.8）
- [x] 坐标对齐（OpenCV→Y-up）+ scale recovery
- [x] YOLO+SAM2 → 3 objects, real COCO labels (vs 69 auto)
- [x] 动态 BEV grid + Nav2 costmap
- [x] 消融实验（4 变体 × 3 帧配对）— 2026-06-19 复现确认
- [x] 三方向文献调研完成

**Baseline 结论**：fused grid: FG/BG IoU=0.052, BEV coverage 1.88%（动态 grid 的 85.5% 是压缩 grid 范围的假象），costmap 55% lethal, 75% drivable conflict。三个机制性失败阻止 BEV 可用——见 `docs/current_issues.md`。

### Phase B：ReSplat Decoder Retraining（当前）🔜

> 完整设计见 `docs/lit_notes/phaseb_design_2026-06-19.md`

- [x] **Phase A.1 POC** (2026-06-19): post-hoc MLP 无效 → 确认端到端重训必要性
- [x] 损失函数严谨数学推导（概率占据场→NLL→Chamfer+Focal+Hinge+CE+L1）
- [x] 后端选型分析：首选 ReSplat (16× fewer Gaussians)，备选 CoSplat
- [x] Teacher 选型：VGGT-Ω (CVPR 2026 Oral, depth δ1.25=93.5%)
- [x] 超参优化策略：Optuna → PBT → BO；RL 用于高斯密度分配
- [x] 获取 ReSplat 代码 (github.com/cvg/ReSplat, MIT, cc4594a)
- [x] 获取 VGGT-Ω 代码 (github.com/facebookresearch/vggt-omega, 39a0cb8)
- [x] 实现 ReSplat wrapper (`eof3r/src/foreground/resplat_wrapper.py`)
- [x] 实现 occupancy head + semantic head (`eof3r/src/training/heads.py`)
- [x] 实现损失函数 (`eof3r/src/training/losses.py`): L_depth + L_occ + L_free + L_sem + L_color
- [x] 实现 VGGT 监督标注 (`eof3r/src/training/supervision.py`): 逐高斯投影标记
- [x] 实现 VGGT 监督预计算脚本 (`eof3r/scripts/preprocess/precompute_vggt_supervision.py`)
- [x] 实现三阶段训练器 (`eof3r/src/training/trainer.py`)
- [x] 实现训练脚本 (`eof3r/scripts/train/train_phase_b.py`)
- [ ] 申请 VGGT-Ω checkpoint 访问权限 (HuggingFace gated)
- [ ] 设置 ReSplat 独立 conda 环境 (Python 3.12 + PyTorch 2.7.0 + CUDA 12.8)
- [ ] 运行 VGGT 监督预计算 on Re10k
- [ ] 运行三阶段训练 on Re10k
- [ ] 验证：对比 baseline opacity vs retrained occupancy BEV 质量

### Phase C：可微 BEV + Free-Space Carving
- [ ] 将 numpy BEV 替换为 torch 可微操作
- [ ] 解析 Σ→XZ 投影（保留协方差结构）
- [ ] BEV CUDA kernel 加速

### Phase D：端到端 Planning Loss
- [ ] costmap 质量指标作为可微损失
- [ ] RL 自适应高斯密度分配（RLD-GS 风格）
- [ ] 三消融论文核心结果

---

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
