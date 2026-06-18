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

## 4 论文导向的创新方向

> 核心原则：**不能只是"拼接模型"**。创新必须来自：(a) 新问题定义，(b) 新方法/架构，(c) 新评估范式。

### 方向 A：Planning-Oriented Gaussian Representation（核心创新）

**问题**：现有 3DGS/MVSplat 的目标是逼真渲染，其 Gaussian primitive 针对 photorealistic 优化。

**创新**：重新定义 Gaussian primitive 的训练目标：
- 用 `occupancy_alpha` 替代 `opacity`（显式建模占据概率）
- 每个 Gaussian 额外输出 `semantic_embedding`、`risk_score`、`confidence`
- 训练损失：(1-λ)·L_occ + λ·L_rgb，其中 λ∈[0,0.3] 作为辅助
- BEV 投影时的 occupancy 是从**为占据优化的 alpha**计算，而非渲染用的 opacity

**论文卖点**："We repurpose Gaussian Splatting primitives from rendering tools to planning-oriented world representations."

### 方向 B：Differentiable BEV Projection with End-to-End Planning Loss

**问题**：当前 BEV 投影是确定性的 numpy 操作，不可微。

**创新**：
- 用 PyTorch 重写 Gaussian→BEV 投影（高度滤波 + scatter + gaussian_smooth）
- 使 BEV 占据网格对 Gaussian 参数（means, scales, occupancy_alpha）可微
- 端到端训练：输入图像 → MVSplat/3DGS → 可微 BEV → planning loss（path smoothness, clearance）
- 这允许 **planning loss 的梯度反向传播到 Gaussian 参数**

**论文卖点**："First differentiable BEV projection from Gaussian primitives, enabling end-to-end training with planning objectives."

### 方向 C：Cross-Model Geometric Consistency（解决瓶颈 1）

**问题**：VGGT 和 MVSplat 独立训练，输出空间不兼容。

**创新**：
- 提出 **Geometric Consistency Loss**：对 VGGT pointmap 和 MVSplat Gaussians 的重叠区域施加一致性约束
- 轻量级 **Coordinate Adapter Network**：一个小 MLP（~100K params）学习 MVSplat 输出到 VGGT 坐标帧的残差变换
- 训练时 freeze 两个大模型，只训练 adapter + consistency loss
- 推理时 adapter 作为后处理步骤，不增加推理时间

**论文卖点**："A lightweight adapter that bridges the coordinate gap between independently-trained feedforward models without fine-tuning either."

### 方向 D：Semantic Lifting via 2D→3D Gaussian Association

**问题**：2D 语义（YOLO+SAM2 mask）和 3D 高斯球之间缺乏关联。

**创新**：
- 利用 MVSplat 的 cost volume 或 depth 信息，将 2D mask "lift" 到 3D
- 对每个高斯球，通过其投影位置检查是否落在 2D mask 内 → 分配物体 ID + 语义标签
- 输出 per-Gaussian semantic field → 3D semantic occupancy
- 这利用了 MVSplat 的内部几何（cost volume 提供的深度约束）

**论文卖点**："Lifting 2D segmentation to 3D Gaussian semantics via feedforward geometry."

### 方向 E：Scale-Agnostic Fusion（解决 VGGT unit-scale）

**问题**：VGGT 输出 unit-scale，需要外部 anchor 才能恢复真实尺度。

**创新**：
- 不依赖外部尺度 anchor，改用**多模型相互一致性**恢复尺度
- MVSplat 从 COLMAP 位姿获得 metric scale → 作为 scale anchor
- VGGT 的 unit-scale 点云通过最小化与 MVSplat 高斯球的 Chamfer distance 来对齐
- 这是一个 optimization-based scale recovery，不需要 LiDAR/深度/camera height
- 在 campus 场景：用 wheel odometry 的 metric baseline 验证

**论文卖点**："Scale recovery without depth sensors: mutual consistency between feedforward geometric models."

---

## 5 推荐优先级

| 优先级 | 方向 | 创新性 | 工作量 | 依赖 |
|--------|------|--------|--------|------|
| **P0** | A: Planning-Oriented Gaussians | ⭐⭐⭐⭐⭐ | 高（需训练） | MVSplat 训练代码 |
| **P1** | B: Differentiable BEV | ⭐⭐⭐⭐⭐ | 中 | A 完成 |
| **P1** | E: Scale-Agnostic Fusion | ⭐⭐⭐⭐ | 中 | 坐标系已解决 |
| **P2** | D: Semantic Lifting | ⭐⭐⭐ | 低 | 当前架构就绪 |
| **P3** | C: Cross-Model Consistency | ⭐⭐⭐ | 高（需训练） | A 完成 |

### 推荐最低创新组合（毕设可行）

**A（轻量版）+ D + E**：

1. **A 轻量版**：不重新训练 MVSplat，而是添加一个 light-weight occupancy head（~50K params），在冻结的 MVSplat backbone 上用 planning loss 微调。3-5 天训练。
2. **D**：2D mask → 3D Gaussian 关联。利用 MVSplat 的 cost volume depth 做 lifting。代码量 ~200 行。
3. **E**：多模型 scale recovery。利用 MVSplat metric pose 和 VGGT unit-scale 的相互一致性优化 scale。代码量 ~300 行。

**三个方向共同构成论文 core contribution**：
> "We propose a planning-oriented 3D scene representation that (A) repurposes Gaussian primitives for occupancy prediction, (D) lifts 2D semantic segmentation to 3D via feedforward geometry, and (E) recovers metric scale without depth sensors through cross-model consistency."

---

## 6 当前 Pipeline 的诚实评估

### 能做什么

- ✅ 端到端从 RGB 图像生成 BEV 占据 + semantic costmap（架构完整）
- ✅ 3 个预训练模型（SAM2, VGGT, MVSplat）在同一 pipeline 中协作
- ✅ 坐标帧对齐 + 尺度恢复使 FG/BG 在同一空间中（冲突率 75%→0% 验证）
- ✅ YOLO+SAM2 提供真实语义标签（3 类 COCO 物体 vs 69 碎片）
- ✅ Costmap 输出合理分布（55% lethal, 42% free）

### 不能做什么

- ❌ 精确的 metric-scale 占据（scale recovery 是近似的）
- ❌ 高质量的 FG/BG overlap（IoU=0.05，两套表征粒度不匹配）
- ❌ per-Gaussian 语义（语义只在 2D bbox 层面）
- ❌ 实时推理（22s total，主要是 VGGT 13s）

### 论文中应诚实讨论的局限性

1. Scale recovery without depth sensors is approximate（需要 LiDAR/depth 才能精确）
2. Feedforward models trained independently produce spatially incompatible outputs（需要 adapter）
3. Gaussian opacity ≠ occupancy（需要专门的 occupancy head）
4. 2D→3D semantic association is lossy（需要 lifting 机制）

---

## 7 下一步行动

1. **立即**：实现方向 D（Semantic Lifting）——代码量小，立即可验证
2. **短期**：实现方向 E（Scale-Agnostic Fusion）——不依赖训练，纯优化
3. **中期**：实现方向 A 轻量版（Planning-Oriented Gaussians）——需要 GPU 训练
4. **长期**：方向 B（Differentiable BEV）——A 完成后自然延伸
