# 三方向文献调研综合报告

> 日期：2026-06-18 | 为 EOF3R 论文创新方向选型

---

## 方向 1：MVSplat ↔ VGGT 跨模型表征对齐

### 核心挑战
两个独立训练的 feedforward 模型在相同坐标帧中仍产生空间不兼容的输出（IoU=0.05）。

### 调研发现

| 方法 | 会议 | 与我们的相关性 | 代码 |
|------|------|---------------|------|
| **CSG-Fusion** | ICCV 2025W | ⭐⭐⭐⭐⭐ MASt3R pointmap → Gaussian，跨视图一致性 | 待确认 |
| **GaussianCross** | MM 2025 | ⭐⭐⭐ 尺度不一致点云→统一 Gaussian 表征 | HuggingFace |
| **SIREN** | CoRL 2025 | ⭐⭐⭐ 3DGS map-to-map 注册，无需初始位姿 | 待确认 |
| **UniPre3D** | CVPR 2025 | ⭐⭐ 点云预训练，跨模态 Gaussian | 学术 |

### 推荐路径：CSG-Fusion 风格 + 轻量 Adapter

**不做**：重新训练 MVSplat 或 VGGT（太昂贵）

**做**：
- 利用 VGGT pointmap 作为"几何脚手架"引导 MVSplat 高斯球的位置
- 训练一个轻量 Pointmap→Gaussian Offset Network（~50K params, MLP）
- 输入：VGGT pointmap 的局部几何特征
- 输出：对 MVSplat 高斯球 means 的残差修正
- 损失：渲染深度与 VGGT pointmap 的 Chamfer Distance + 跨视图一致性

**创新表述**：*"A lightweight geometry adapter that bridges the representation gap between independently-trained feedforward models, using pointmap-guided Gaussian refinement without fine-tuning either model."*

---

## 方向 2：Planning-Oriented 前馈高斯（替代 Photorealistic 高斯）

### 核心挑战
MVSplat 的高斯球为渲染优化（opacity、SH 系数），非为占据预测优化。

### 调研发现

| 方法 | 会议 | 关键创新 | 代码 | 可直接用？ |
|------|------|---------|------|-----------|
| **GaussianOcc** | ICCV 2025 | 自监督占据估计，voxel→Gaussian splatting，5× 加速 | ✅ GitHub | ✅ 最相关 |
| **GaussianFormer** | ECCV 2024 | 首次用 Gaussian 做 3D 语义占据预测 | ✅ GitHub | ✅ 已在 lit notes |
| **GaussRender** | ICCV 2025 | 可微 Gaussian 渲染改进占据 | ✅ GitHub | ⚠️ 辅助方法 |
| **DrivingGaussian** | CVPR 2024 | 复合 Gaussian Splatting 自动驾驶场景 | ✅ GitHub | ⚠️ 全场景重建 |

### 推荐路径：GaussianOcc 架构 + 我们的 feedforward 前端

**GaussianOcc 的核心贡献**（可直接复用）：
1. **GSP (Gaussian Splatting for Projection)**：自监督训练，无需 GT 6D 位姿
2. **GSV (Gaussian Splatting from Voxel)**：从 voxel 直接 splat Gaussian，替代慢速 volume rendering

**我们的改进**：
- GaussianOcc 用 surround-view 相机 + 自监督，我们改用 feedforward MVSplat 作为 Gaussian 初始化
- 添加 occupancy head（替代 SH 颜色头）：每个高斯球输出 `[occupancy_alpha, semantic_class, confidence]`
- 训练目标：L_occ + L_footprint + L_semantic + λ·L_rgb（λ=0.1）

**创新表述**：*"We extend feedforward 3DGS with planning-oriented Gaussian primitives that directly predict occupancy, semantics, and confidence — replacing photorealistic opacity and SH coefficients with navigation-relevant parameters."*

---

## 方向 3：2D 语义 → 3D 高斯球 / 点云级精确语义

### 核心挑战
YOLO+SAM2 的 2D mask 无法关联到 MVSplat 的 131K 高斯球。

### 调研发现

| 方法 | 会议 | 关键创新 | 代码 | 训练需求 | 适合我们？ |
|------|------|---------|------|---------|-----------|
| **Gaussian Grouping** | ECCV 2024 | Identity Encoding per Gaussian，SAM 监督 | ✅ Apache 2.0 | ~1h/scene | ⭐⭐⭐⭐⭐ |
| **SAGA** | arXiv 2023 | 高斯亲和特征 + 对比学习，点提示 GUI | ✅ | ~10K iter | ⭐⭐⭐⭐ |
| **LangSplat** | CVPR 2024 | SAM+CLIP→3D 语言高斯，199× LERF | ✅ | 需自编码器 | ⭐⭐⭐ |
| **SA3D** | NeurIPS 2023 | SAM→NeRF 桥，单视图提示→3D mask | ✅ | ~2min | ⭐⭐⭐ |
| **SAGS** | 2024 | SAGA 改进：边界增强 + Grounding-DINO 文本提示 | ✅ | ~10K iter | ⭐⭐⭐ |

### 推荐路径：Gaussian Grouping（最直接适配）

**为什么选 Gaussian Grouping 而非 SAGA**：
1. GG 的 Identity Encoding 是每个高斯球的**固定属性**（类似物体 ID），推理时无需交互
2. SAGA 需要点提示交互（不适合自动驾驶全自动场景）
3. GG 的 SAM 监督流程与我们的 YOLO+SAM2 前端天然兼容
4. GG 训练快（~1h），Apache 2.0 开源

**适配方案**：
- 将 GG 的 Identity Encoding 集成到 MVSplat 的 decoder 中
- 用 YOLO 检测框（而非 SAM "everything"）提供 2D mask 监督
- 输��：per-Gaussian object_id + semantic_class + confidence
- 推理时：通过 object_id 聚类→每个物体一组高斯球→计算 BEV footprint

**创新表述**：*"We lift 2D detection-level semantics to 3D Gaussian primitives via per-Gaussian identity encoding, enabling object-level Gaussian occupancy prediction without per-scene optimization."*

---

## 综合推荐：可立即尝试的方法

### 优先级 P0（本周可试，代码已有）

**1. Gaussian Grouping → 2D→3D 语义 Lifting**
```bash
git clone https://github.com/lkeab/gaussian-grouping.git
# 研究其 Identity Encoding 机制
# 适配到 MVSplat decoder 输出
```
- 代码量：~300 行改动
- 验证方式：训练后检查 per-Gaussian 的 object_id 精度

**2. GaussianOcc → Planning-Oriented 占据**
```bash
git clone --recurse-submodules https://github.com/GANWANSHUI/GaussianOcc.git
# 研究 GSV (Gaussian Splatting from Voxel) 模块
# 替换我们的 numpy BEV 投影
```
- 代码量：~500 行集成
- 验证方式：BEV coverage 定量对比（GaussianOcc vs 当前）

### 优先级 P1（需 1-2 周训练）

**3. 轻量 Pointmap→Gaussian Adapter**
- 训练数据：Re10k 场景（VGGT pointmap + MVSplat Gaussians）
- 网络：3 层 MLP，~50K 参数
- 损失：Chamfer Distance + cross-view consistency
- 预期：FG/BG IoU 从 0.05 提升到 0.15+

### 优先级 P2（完整论文 contribution）

**4. End-to-End Planning-Oriented 3DGS**
- 组合 1+2+3
- 在 MVSplat decoder 上添加 occupancy/semantic/confidence head
- 用 GaussianOcc 的 GSV 做可微 BEV 投影
- 用 Gaussian Grouping 的 Identity Encoding 做语义
- 用 Adapter 对齐 VGGT+MVSplat

---

## 关键论文的代码可用性

| 论文 | GitHub | Stars | License | 环境要求 |
|------|--------|-------|---------|---------|
| Gaussian Grouping | `lkeab/gaussian-grouping` | 391 | Apache 2.0 | Python 3.8, CUDA |
| GaussianOcc | `GANWANSHUI/GaussianOcc` | new | — | Python 3.8, CUDA 11.3 |
| GaussianFormer | `Huang-yuanhui/GaussianFormer` | — | — | — |
| SAGA | `Jumpat/SegAnyGAussians` | — | — | Python 3.8, SAM ViT-H |
| LangSplat | `minghanqin/LangSplat` | — | — | SAM + CLIP |

---

## 三方向与论文贡献的映射

```
论文核心贡献：
┌─────────────────────────────────────────────┐
│ Planning-Oriented Feedforward 3D Scene      │
│ Representation for Robot Navigation         │
├─────────────────────────────────────────────┤
│ 1. Cross-Model Geometric Adapter (方向1)    │
│    → 解决独立训练模型的表征鸿沟              │
│                                             │
│ 2. Occupancy-First Gaussian Primitives (方向2)│
│    → 将渲染用高斯改造为规划用占据表示        │
│                                             │
│ 3. 2D-to-3D Semantic Lifting (方向3)        │
│    → per-Gaussian 语义 → object级占据预测   │
└─────────────────────────────────────────────┘
```

三个方向相互支撑：Adapter 确保 FG/BG 几何一致 → Occupancy head 确保占据精度 → Semantic lifting 确保语义完整。最终产出：**一个从 RGB 图像端到端预测 planning-oriented 3D 占据+语义的前馈系统**。
