# EOF3R Pipeline 综合评估与机制分析

> 日期：2026-06-19 | 消融实验完成 | 方向更新：跨模型几何蒸馏
>
> **定位更新**（2026-06-19）：消融实验揭示了"拼接预训练模型"的根本局限——三个机制性失败（opacity≠occupancy, covariance loss, no free-space model）无法通过调参解决。项目方向从"串行拼接 VGGT+MVSplat"转为"跨模型几何蒸馏"（VGGT 作为训练时的 teacher，MVSplat 作为推理时唯一模型）。详见 §2 机制分析。

---

## 1 定量结果

### 1.1 消融实验总表

| 变体 | BEVcov | FG/BGIoU | Conflict | Free% | Lethal% | Obj | Sem | Scale | Time |
|------|--------|----------|----------|-------|---------|-----|-----|-------|------|
| **A_full** | 0.855 | **0.052** | **0.754** | 0.417 | 0.550 | **3** | **Y** | 7.8 | 1.1s |
| B_noscale | 0.543 | 0.000 | 0.000 | 0.616 | 0.345 | 3 | Y | 7018 | 0.7s |
| C_noalign | 0.817 | 0.000 | 0.000 | 0.479 | 0.489 | 3 | Y | 7.8 | 0.7s |
| D_auto | 0.855 | 0.052 | 0.754 | 0.417 | 0.550 | 69 | N | 7.8 | 2.7s |

> 注：BEVcov 使用动态 grid (auto bounds)，85% 的覆盖率是自适应假象。固定 grid (400×227) 下 coverage(t=0.3)=**1.88%**，IoU=**0.0047**，conflict=**0.418**。动态 grid 将空间范围压缩到 ~4.9m²（真实场景 floor area ~20-30m²），详见 §2.4。

### 1.2 各组件贡献度

| 组件 | 消融对比 | 核心贡献 |
|------|---------|---------|
| Scale Recovery | A vs B | FG/BG IoU 0→0.052（从零到有 overlap），scale 从 ~7000 纠正到 7.8 |
| Coord Alignment | A vs C | Drivable conflict 0→75%（FG 物体正确落在 BG 可通行区域上） |
| YOLO Frontend | A vs D | 69→3 objects，语义标签 Y/N，2.5× 加速 |
| Dynamic BEV | — | Grid 自适应数据范围（400×227 vs 固定 1600×1600） |

---

## 2 机制分析

### 2.0 三个机制性失败模式（根本原因）

消融实验不仅给出了定量结果，更揭示了"拼接预训练模型"方案的三个结构性失败——它们不是调参问题，而是**类别错误**。

**失败模式 1：Opacity ≠ Occupancy**
MVSplat 的 opacity 与 SH 颜色在 alpha-blending 渲染方程中联合优化：低 α+高 c 与高 α+低 c 可产生相同像素。opacity 是"对渲染颜色的相对贡献权重"，不是"该空间位置的占据概率"。BEV 投影时把 α=0.28 当 28% 占据——类别错误。

**失败模式 2：协方差结构丢失**
Scatter+smooth 用各向同性高斯核替代每个高斯球的 3×3 协方差 Σ。10cm 宽的椅腿高斯球被 `3·max(scale)` 扩张为各向同性圆形——严重过膨胀。

**失败模式 3：无自由空间建模**
VGGT pointmap 表面点全部当作"占据"→ BEV。但每条 VGGT 像素光线隐含 free-space 信息：相机→表面=FREE，表面附近=OCCUPIED，表面后方=UNKNOWN。不做 carving → costmap 无法区分 free 和 unknown。

**解决方案**：跨模型几何蒸馏——用 VGGT 的 depth/pointmap/rays 作为训练监督，重新训练 MVSplat 的 decoder head（occupancy + semantic + confidence），而非拼接两个模型的推理输出。详见 `CLAUDE.md §1c` 和 `current_issues.md`。

---

### 2.1 尺度恢复为什么有效——但不完美

### 2.1 尺度恢复为什么有效——但不完美

**机制**：VGGT 训练时做了两件事：
1. 将世界帧平移到首帧相机原点
2. 除以 `avg_scale = mean(||world_points||)`（所有有效点到首帧相机的平均距离）

推理时 VGGT 输出的是 unit-scale 坐标。我们用地平面估计反推 `avg_scale`：相机距地平面距离为 `d_vggt`（unit-scale），若真实相机高度为 1.5m，则 `scale = 1.5 / d_vggt = 7.8`。

**有效但不完美的原因**：
- VGGT 的地平面估计本身有误差（SVD 拟合粗糙）
- 真实相机高度是人工假设（1.5m），Re10k 数据实际高度未知
- Unit-scale 归一化同时缩放 + 旋转了场景（非均匀 scaling）
- 结论：`scale=7.8` 将坐标恢复到了近似米制，但非精确校准

### 2.2 FG/BG Overlap IoU 为什么只有 0.05

这是当前 pipeline **最根本的机制性问题**。IoU=0.05 意味着 MVSplat 高斯球和 VGGT pointmap 在 BEV 投影后几乎不重叠。

**机制分析**——三个独立原因共同导致：

**(a) MVSplat 使用真实 Re10k 图像但 VGGT 估计的位姿**：
- MVSplat 训练在 COLMAP 位姿（metric scale，特定 baseline）
- 推理时我们喂 VGGT 的 unit-scale→scaled 位姿
- VGGT 的位姿估计误差会传递到 MVSplat 的几何预测中
- MVSplat 的 cost volume 依赖精确的相对位姿来三角化深度

**(b) MVSplat 预测的几何是稀疏的**：
- 131K 高斯球看起来很多，但 MVSplat 将它们集中在 cost volume 有纹理响应的区域
- Re10k 256×256 低分辨率输入限制了场景细节
- 高斯球大多在"物体边缘"而非"物体表面"，导致 BEV 投影稀疏

**(c) BG pointmap 的 BEV 投影方式过于粗糙**：
- 当前将 VGGT pointmap 的每个 3D 点当作等权高斯球（opacity=0.3-0.8, scale=0.1m）
- 14 万个点 → 投影到 BEV → 覆盖大部分地面区域
- FG 的 88K 高斯球（高度滤波后）→ 覆盖几个小区域（家具）
- BG 覆盖大区域、FG 覆盖小区域 → IoU 自然低

**本质问题**：这不是"没对齐"，而是两套表征的**空间粒度不匹配**——BG 是密集 pointmap，FG 是稀疏 Gaussian。

**(d) 补充证据——Drivable Conflict = 75%**：
这个看似"负面"的指标实际上证明了对齐是正确的。FG=occupied 且 BG=drivable 意味着**前景物体坐落在可通行地面上**——这正是物理上预期的。75% 的冲突率说明 MVSplat 的高斯球正确地落在了 BG 地平面之上。

### 2.3 无对齐时为什么 Conflict = 0

变体 C（无坐标对齐）中 FG/BG IoU = 0 且 Conflict = 0。这是因为 MVSplat 用合成位姿（相机 0 在原点看 +Z，相机 1 在 (0.3, 0, 0)），而 VGGT 自估计位姿在完全不同的帧中。两套三维输出在空间中**毫无交集**。

这证明两个模型如果不加协调直接输出，**不可能拼接**——坐标对齐是强制性的前置条件。

### 2.4 BEV Coverage 85% 的误导性

动态 BEV grid（auto bounds）将 grid 收缩到数据实际范围（~几米），使得 coverage 看起来很高（85%）。但这是**自适应的假象**——coverage 高只是因为 grid 变小了。

**真实的 spatial extent 只有 4.9m²**，而真实场景的 floor area 可能是 20-30m²。VGGT unit-scale 压缩了场景尺度，scale recovery 只恢复了一个近似因子。

---

## 3 根因总结：卡在什么地方

从机制层面看，当前 pipeline 有三个**结构性瓶颈**，不是调参能解决的：

### 🔴 瓶颈 1：两套独立训练的模型没有联合优化

MVSplat 和 VGGT 是在不同数据集、不同损失函数、不同坐标约定下独立训练的。即使我们手工对齐了坐标帧和尺度，**内部表征的空间分布仍然不兼容**：
- MVSplat 学习的是"给定稀疏视图和精确位姿，预测场景几何"
- VGGT 学习的是"给定多帧图像，联合估计位姿和场景几何"
- 当 MVSplat 收到 VGGT 的位姿（带误差），其几何预测质量退化

**这不是对齐问题，是分布偏移（distribution shift）问题。**

### 🔴 瓶颈 2：从"photorealistic 输出"到"规划导向表示"的鸿沟

MVSplat 设计目标是 novel view synthesis（逼真渲染），其高斯球包含：
- SH 系数（颜色、view-dependent effects）→ 对规划无用
- Opacity（alpha 通道）→ 部分有用
- Covariance/scale（几何形状）→ 有用但非直接优化

当前 `extract_occupancy()` 对每个高斯球的所有维度做等权处理，没有专门的 occupancy head。**高斯球的 opacity 是针对渲染优化的，不是针对占据预测优化的。**

### 🔴 瓶颈 3：语义信息流不完整

YOLO→SAM2 在 2D 层面产生了语义标签，但这个信息没有沿着 pipeline 流下去：
- 2D mask → 3D Gaussian 之间缺乏关联（哪个 Gaussian 属于哪个物体？）
- MVSplat 输出 131K 个高斯球，但没有 per-Gaussian 的物体 ID
- 语义 BEV 用 bbox 中心点粗略投影，丢失了物体形状信息

---

## 4 Phase A.1: Occupancy Head POC（2026-06-19）

### 实验设计

在 Phase A 证明 opacity ≠ occupancy 之后，测试最直接的修复方案：**用 VGGT depth 监督训练一个 post-hoc MLP，将 Gaussian opacity 替换为预测的 occupancy**。

### 关键结果

| 指标 | Opacity | VGGT Labels | MLP Predicted |
|------|---------|-------------|---------------|
| BEV cov (>0.3) | 0.11% | 0.006% | 0.09% |
| 占据 cells (>0.3) | 2,493 | 140 | 2,016 |
| Density | 0.43 | 0.84 | 0.34 |

### 失败分析

- VGGT depth 投影标记：仅 2.6% Gaussians 在 VGGT 表面（±0.3m），68.9% 在自由空间
- MLP 收敛（val acc 96.5%）但无法提升 BEV——**所有方法 coverage <1%**
- **根因**：不是 opacity 值的问题，而是 **Gaussian 位置本身不对**（MVSplat decoder 为渲染优化，非几何准确）
- **结论**：post-hoc MLP 路线不可行 → Phase B 必须端到端重训 decoder + Gaussian adapter，几何 loss 反向传播至 μ, Σ

---

## 5 后续方向：Phase B 统一方案

> 详见 `docs/lit_notes/phaseb_design_2026-06-19.md`

所有五个旧方向（A/B/C/D/E）已统一为 **跨模型几何蒸馏** 单框架：

| 旧方向 | 在新框架中的对应 |
|--------|----------------|
| A: Occ Head | $\mathcal{L}_{\text{occ}}$ (Focal Loss) + 端到端重训 Gaussian positions |
| B: Diff BEV | Phase C：可微 BEV 边缘化 + CUDA kernel |
| C: Cross-Model Consistency | $L_{\text{depth}}$ (Chamfer) 直接对齐 Gaussian 与 VGGT |
| D: Semantic Lifting | $\mathcal{L}_{\text{sem}}$ + per-Gaussian identity encoding |
| E: Scale Recovery | VGGT-Ω 的 metric-scale 预训练（待验证） |

**核心区别**：旧方案是"轻量级 patch"，Phase B 是**系统性的端到端几何蒸馏**——损失函数经概率模型严谨推导，训练策略分三阶段。
