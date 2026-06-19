# Phase B Deep-Dive: Five Critical Questions

> 日期：2026-06-19 | 调研范围：VGGT精度、锚点方法、自由空间建模、损失函数设计、训练/推理效率
> 触发：Occupancy Head POC 实验证明 post-hoc MLP 不够 → 需要系统性地设计 Phase B

---

## Q1: VGGT 几何先验的精度——能否作为可靠的训练监督？

### VGGT 的定量精度

VGGT（CVPR 2025 Best Paper）在多个基准上的深度估计精度：

| 数据集 | 场景类型 | AbsRel ↓ | δ<1.25 ↑ |
|--------|---------|----------|----------|
| Bonn | 真实室内 | 5.5% | 97.1% |
| KITTI | 真实室外 | 7.2% | 93.8% |
| NYU-v2 | 真实室内 | 6.0% | 95.1% |
| Sintel | 合成动态 | 27.6% | 67.5% |
| ETH3D | 真实扫描 | SOTA | SOTA |

**关键结论**：
- 在真实室内/室外场景：**AbsRel 5-7%**——10m 处误差约 0.5-0.7m，可接受
- StereoBench 评估：VGGT 在真实数据上**显著优于**专用立体匹配网络（FoundationStereo，AbsRel 29% vs 7-10%）
- 主要弱点：**metric-scale 需要外部 anchor**（Scale Ratio 漂移明显）

### VGGT-Ω（CVPR 2026 Oral）——精度飞跃

| 指标 | VGGT | VGGT-Ω | 提升 |
|------|------|--------|------|
| Sintel 深度 δ1.25 | 74.1% | **93.5%** | +26% |
| Sintel 相机 AUC@3° | 22.5 | **40.0** | +77% |
| 推理速度 | 1× | **1.6×** | — |
| 训练内存 | 100% | **30%** | -70% |
| 训练数据 | 数万序列 | **400万序列** | 15× |

**对我们的影响**：应该以 VGGT-Ω 为目标监督模型（精度 93.5% δ1.25，远超原版 VGGT）。原版 VGGT 在我们场景中的 5-7% AbsRel 已有一定实用性，VGGT-Ω 的精度接近实用级别。

### 校园场景的适用性评估

| 场景特征 | VGGT/VGGT-Ω 适用性 |
|---------|-------------------|
| 室外自然光 | ✅ KITTI 验证 (AbsRel 7.2%) |
| 结构化环境（建筑、道路） | ✅ 特征丰富，cost volume 受益 |
| 动态物体（行人、车辆） | ⚠️ VGGT-Ω 训练数据含 1/3 动态内容，鲁棒性提升 |
| 低纹理区域（白墙、路面） | ⚠️ 纯 cost-volume 弱，Depth Anything 融合有帮助 |
| 尺度漂移 | ❌ 需要外部 anchor（已知相机高度、LiDAR 点、轮式里程计） |

**诚实结论**：VGGT-Ω 的几何先验在有外部尺度 anchor 的情况下**足够作为训练监督**（不是 GT 级别，但是够好的 teacher）。尺度恢复仍然需要外部信息。

---

## Q2: 锚点方法——只用锚点是否足够？有没有更好的几何表示？

### 锚点流派综述

| 方法 | 锚点来源 | 高斯生成 | 几何精度机制 |
|------|---------|---------|-------------|
| **Scaffold-GS** (CVPR 2024) | SfM 点云 → 体素化 | Anchor feat + MLP → Gaussians | 结构约束防止漂浮 |
| **HAC** (ECCV 2024) | 继承 Scaffold-GS | 同 Scaffold-GS | Hash-grid 辅助压缩 |
| **HAC++** (TPAMI 2025) | 继承 Scaffold-GS | 同 Scaffold-GS | 自适应 offset masking 剪枝 |
| **AnchorSplat** (CVPR 2026) | MVS 点云 → FPS 采样 | Transformer → per-anchor Gaussians | feedforward 版锚点 |
| **Octree-GS** | 八叉树层级锚点 | LOD 层级生成 | 多尺度几何约束 |
| **GaussianFormer** (ECCV 2024) | 无显式锚点 | 3D query → cross-attn → Gaussians | 隐式位置编码 |

### 锚点的根本局限——以及我们的机会

**锚点方法擅长**：消除 floaters（Gaussians 被约束在锚点附近）、压缩存储、加速渲染

**锚点方法不擅长**：精确表示物体几何尺度。原因：
- 锚点是**点**——没有体积信息。锚点"附近"生成 Gaussians → 无法区分 "10cm 宽的椅腿" 和 "1m 宽的沙发扶手"
- 锚点不携带**自由空间**信息——不知道锚点"前方"是空的

**更好表示几何尺度的方式**：

| 方式 | 代表方法 | 几何信息量 | 计算开销 |
|------|---------|----------|---------|
| 稀疏锚点 | Scaffold-GS, HAC | 低（只有位置） | 低 |
| **3D 占据体素** | **GaussianOcc** | **高（体积占据）** | 中 |
| G-Splat 平面 | PGSR | 高（平面先验） | 中 |
| Signed Distance | GSDF | 最高（SDF 隐式表面） | 高 |
| 稠密 pointmap | VGGT | 高（dense 表面点） | 免费（已有） |

**我们的最佳方案**：**不是纯锚点，而是 VGGT pointmap(表面) + 光线(自由空间) → 联合约束 MVSplat 的 Gaussian 生成。**

锚点只是表面约束——VGGT pointmap 天然提供了稠密的表面锚点。我们独有的是**自由空间光线**（相机→表面=自由，表面=占据，表面后=未知）——这是所有现有锚点方法都缺少的。

---

## Q3: VGGT 光线三值标记（FREE/OCCUPIED/UNKNOWN）——难度多大？需要重训 VGGT 吗？

### 答案：不需要重训 VGGT。这完全是一个后处理操作。

VGGT 已经输出了做 free-space carving 所需的所有信息：
1. **每像素深度**（从 pointmap 的 Z 坐标计算）
2. **相机位姿**（9D 编码 → camera-from-world 矩阵）
3. **相机内参**（从 FOV 和分辨率推算）

### 光线三值标记算法（纯后处理，零训练）

```
对 VGGT 第一帧的每个像素 (u, v)：
  depth_vggt = pointmap_cam[u, v, 2]       # Z in camera frame
  
  # 在 3D 空间中采样光线上的点
  for d in linspace(0.1, far_clip, N_samples):
    P_3d = camera_center + d * ray_direction(u, v)
    
    if d < depth_vggt - σ:          # 表面前方 → FREE
      label[P_3d] = FREE
    elif abs(d - depth_vggt) <= σ:  # 表面附近 → OCCUPIED
      label[P_3d] = OCCUPIED
    else:                            # 表面后方 → UNKNOWN
      label[P_3d] = UNKNOWN
```

**计算开销**：对 280×504 点云 × 每条光线 100 个采样点 = ~14M 个标记点。纯 numpy，约 0.1s。

### 但关键问题是：如何将这些标记转化为 MVSplat 的 Gaussian 训练监督？

这是我们 POC 实验已经解决的问题：**深度投影标记法**
- 对每个 MVSplat Gaussian，投影到 VGGT 相机 → 比较深度 → 标记 free/occ/unknown
- POC 中 71.5% 的 Gaussians 可被标记（68.9% free, 2.6% occupied, 28.5% unknown）

### VGGT → MVSplat 监督的具体流程

```
训练时（每个 batch）：
  1. RGB图像 → VGGT → 深度图 + 位姿（teacher，冻结，不反向传播）
  2. 同一 RGB图像 → MVSplat → Gaussians（student，需要训练）
  3. 对每个 Gaussian：
     a. 投影到 VGGT 相机帧
     b. 比较深度 → 标记 FREE/OCCUPIED/UNKNOWN
  4. 损失反传：
     - 被标记为 OCCUPIED 的 Gaussian：鼓励高 occupancy + 靠近 VGGT 表面
     - 被标记为 FREE 的 Gaussian：鼓励低 occupancy（接近 0）
     - 被标记为 UNKNOWN 的：mask 掉，不参与损失
```

**不需要重训 VGGT**——它是 frozen teacher，只需要 inference。

---

## Q4: 损失函数设计的合理性

### 提出的损失函数

```
L_total = α·L_depth + β·L_occ + γ·L_free + δ·L_semantic + λ·L_color
         (α=1, β=1, γ=1, δ=0.5, λ=0.1)
```

### 各项的合理性分析

#### L_depth（深度损失）
```
L_depth = 1/N Σ Chamfer(P_gaussian, P_vggt)  # Chamfer distance
        + 1/N Σ Huber(d_gaussian, d_vggt)     # Per-pixel depth Huber
```
- **合理**：PM-Loss 已验证 Chamfer + Umeyama alignment 有效（+2dB PSNR）
- **风险**：VGGT 深度误差（5-7% AbsRel）会传播。需要 Huber loss 的抗差性
- **VGGT-Ω 降低风险**：δ1.25 从 74% → 93.5%

#### L_occ（占据损失）
```
L_occ = BCE(σ(occupancy_head(g)), y_vggt_occ)  # Binary CE
```
- **合理**：标准二分类损失，GaussianOcc 成功使用
- **风险**：类别极度不平衡（POC 中 3.6% occ vs 96.4% free）→ 需要 pos_weight 或 focal loss
- **改进**：用 **Focal Loss**（γ=2）替代 BCE，自动处理类别不平衡

#### L_free（自由空间损失）
```
L_free = 1/|FREE| Σ max(0, occ(g) - ε)²   # 自由空间中的 Gaussian 应低占据
       + 1/|FREE| Σ max(0, ε - occ(g))²   # Hinge: 低于 ε=0.1 即可，不强制 0
```
- **创新点**：这是我们的核心创新——现有方法都没有显式的 free-space 监督
- **合理**：hinge loss 比 BCE 更温和——"接近 0 就够"，不强制等于 0
- **风险**：如果 free-space 标记有噪声（VGGT 深度误差导致 free 区域有物体被误标），hinge loss 的鲁棒性优于 MSE

#### L_semantic（语义损失）
```
L_semantic = CE(semantic_head(g), y_sam2_label)  # Per-Gaussian classification
```
- **需要 Gaussian Grouping 风格的 identity encoding 才能实现 per-Gaussian 语义**
- 暂时用 2D mask 投影监督，远期用 per-Gaussian identity + classifier

#### λ·L_color（颜色渲染，辅助）
```
L_color = L1(I_rendered, I_gt) + λ_ssim·(1-SSIM(I_rendered, I_gt))
```
- λ=0.1 是关键——颜色只是"正则化项"，防止网络遗忘 cost volume 的几何提取能力
- 如果 λ 太大（>0.5），优化会退化为 photometric-dominant → Gaussians 又回到渲染优化模式
- 如果 λ=0（完全不要颜色），cost volume 的预训练权重可能漂移太远

### 损失函数总体评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 任务对齐 | ⭐⭐⭐⭐⭐ | 所有 loss term 直接对应规划需求 |
| 梯度稳定性 | ⭐⭐⭐⭐ | BCE + Huber + Hinge 的组合梯度稳定 |
| 类别不平衡处理 | ⭐⭐⭐ | 需要 Focal Loss 替代 BCE |
| 超参数敏感度 | ⚠️ | α/β/γ/δ/λ 五个权重需要调参 |
| 创新性 | ⭐⭐⭐⭐⭐ | L_free 是现有方法都没有的 |
| 计算开销 | ⭐⭐⭐ | 需要 per-Gaussian VGGT 投影（但 O(N) 可并行） |

**最大风险**：多任务损失之间的梯度冲突。L_depth 想移动 Gaussian 到 VGGT 表面，但 L_occ 和 L_free 对位置不敏感（只关心占据值）。解决方案：**分阶段训练**
- Stage 1（warmup）：只用 L_depth + L_color（让 Gaussians 先移到正确位置）
- Stage 2（fine-tune）：加入 L_occ + L_free + L_semantic（训练占据/语义头）

---

## Q5: 训练/推理效率——大模型训练 + 加速技术

### 两个"大模型"的训练挑战

| 模型 | 参数量 | 单次推理时间 | 训练可行性 |
|------|--------|------------|----------|
| VGGT (1B) | ~1B | 13.6s | ❌ 冻结——只做 teacher inference |
| MVSplat | ~100M | 3.6s | ✅ 只训练 decoder + head (~30M) |

**VGGT 不训练**——它是 frozen teacher。训练只针对 MVSplat decoder。

### 可用的加速技术

#### 1. VGGT-Ω 替代原版 VGGT
| 改进 | 影响 |
|------|------|
| 推理 1.6× 加速 | 13.6s → ~8.5s |
| 深度精度 +26% | 更好的监督质量 |
| 训练内存 -70% | 可在单卡上同时跑 VGGT-Ω + MVSplat |

#### 2. AVGGT / Block-Sparse Attention（无需重训）
- AVGGT：100 帧 2× 加速，800 帧 8-10× 加速
- Block-Sparse：最高 4× 加速，无需重训
- **叠加 VGGT-Ω + Block-Sparse → 推理可达 2-3× 总加速**

#### 3. 强化学习 (RL) 的应用

**RL 在 3DGS 中已有的工作**：
- **RLD-GS**（ICONIP 2025）：RL 优化锚点分布——在感知重要区域分配更多 Gaussians
- **RLGS**（arXiv 2025）：RL 自适应调学习率和 densification 阈值

**在我们的场景中 RL 可以用于**：
1. **自适应 Gaussian 密度分配**：在前景/障碍物区域分配更多 Gaussians，背景/自由空间区域少分配（RLD-GS 已验证可行）
2. **超参数自适应**：自动调整 L_depth/L_occ/L_free 的损失权重（RLGS 已验证可行）

**但 RL 不适合**：端到端训练 Gaussian 参数（样本效率太低，不如可微的梯度下降）

#### 4. CUDA 算子重写

**最值得优化的算子**：
| 算子 | 当前 | 可优化到 | 方法 |
|------|------|---------|------|
| Gaussian→BEV 投影 | numpy loop (O(N×r²)) | CUDA scatter (O(N)) | 借鉴 diff-gaussian-rasterization |
| Per-Gaussian VGGT 投影 | numpy vectorized | CUDA kernel | 简单的投影+插值，~50 行 CUDA |
| Occupancy Head MLP | PyTorch (batch) | 已足够快 | 无需优化 |
| Free-space ray 标记 | numpy | GPU parallel | 光线独立，天然并行 |

**优先级**：BEV 投影 CUDA 化 > Per-Gaussian 投影 CUDA 化 > 其他

#### 5. 训练加速

| 技术 | 预期效果 |
|------|---------|
| Mixed Precision (AMP) | 1.5-2× 加速，显存 -40% |
| Gradient Checkpointing | 显存 -50%，速度 -15% |
| Flash Attention (VGGT 中) | 已内置（Swin Transformer） |
| Torch Compile | 1.2-1.5× 加速（PyTorch 2.x） |
| 数据预计算 VGGT 监督 | 离线跑 VGGT → 保存深度+位姿 → 训练时直接加载 |

**推荐策略**：
1. **离线预计算 VGGT 监督**：一次性跑 VGGT-Ω 在所有训练帧上，保存深度图+位姿。训练时直接加载——消除 VGGT 推理开销
2. AMP + Gradient Checkpointing：标准配置
3. BEV 投影 CUDA kernel：中期优化

### 推理速度目标

| 阶段 | 当前 (VGGT + MVSplat) | 优化后 (VGGT-Ω + 优化 MVSplat) | 目标 |
|------|----------------------|-------------------------------|------|
| VGGT 推理 | 13.6s | ~8s (VGGT-Ω) 或 ~4s (VGGT-Ω + AVGGT) | <5s |
| MVSplat 推理 | 3.6s | ~3s (AMP + compile) | <3s |
| BEV 投影 | 0.03s | <0.01s (CUDA) | <0.01s |
| **总计** | **~21s** | **~6-8s** | **<10s** |

6-8s 的总延迟对于低速配送（<0.5 m/s）足够——车在 6s 内移动不到 3m。

---

## 总结：Phase B 技术方案修订

| 原方案 | 修订后方案 | 触发原因 |
|--------|----------|---------|
| 原版 VGGT 作 teacher | **VGGT-Ω** 作 teacher（精度+26%，速度 1.6×） | Q1 |
| 纯锚点约束 Gaussian 位置 | **稠密 VGGT pointmap + 自由空间光线** 联合约束 | Q2 |
| Post-hoc 标记 Gaussians | **训练时在线投影标记**（不需要重训 VGGT） | Q3 |
| BCE Loss | **Focal Loss + Hinge Loss** 处理类别不平衡 | Q4 |
| 单阶段训练 | **两阶段：warmup(几何) → fine-tune(占据+语义)** 防止梯度冲突 | Q4 |
| 在线 VGGT 推理 | **离线预计算 VGGT 监督**（训练时直接加载） | Q5 |
| numpy BEV 投影 | 短期 numpy，中期 **CUDA kernel** | Q5 |

### 需要进一步验证的假设

1. ~~VGGT 深度在校园场景的 AbsRel < 10%~~ → 需要实机 rosbag 验证
2. ~~Focal Loss 能有效处理 3.6% occ vs 96.4% free 的不平衡~~ → 在 POC 实验上快速验证
3. ~~VGGT-Ω 的 400 万序列规模预训练是否覆盖校园类场景~~ → 需要 qualitative 测试

---

## 参考文献

1. Wang et al., "VGGT: Visual Geometry Grounded Transformer," CVPR 2025 Best Paper.
2. Wang et al., "VGGT-Ω," CVPR 2026 Oral, arXiv:2605.15195.
3. Shi et al., "PM-Loss: Revisiting Depth Representations for Feed-Forward 3DGS," 3DV 2026, arXiv:2506.05327.
4. Zhang et al., "AnchorSplat: Feed-Forward 3DGS with 3D Geometric Priors," CVPR 2026, arXiv:2604.07053.
5. Xu et al., "DepthSplat: Connecting Gaussian Splatting and Depth," CVPR 2025.
6. Lu et al., "Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering," CVPR 2024 Highlight.
7. Chen et al., "HAC++: Towards 100X Compression of 3DGS," TPAMI 2025, arXiv:2501.12255.
8. Gan et al., "GaussianOcc: Fully Self-supervised 3D Occupancy Estimation with GS," ICCV 2025.
9. Huang et al., "GaussianFormer: Scene as Gaussians for 3D Semantic Occupancy Prediction," ECCV 2024.
10. Huang et al., "RLD-GS: RL-Driven Gaussian Splatting," ICONIP 2025.
11. Sun et al., "AVGGT: Rethinking Global Attention for Accelerating VGGT," arXiv:2512.02541.
