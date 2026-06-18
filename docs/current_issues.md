# 当前问题与解决方案

> 更新时间：2026-06-19
> 所有真模型（SAM2+VGGT+MVSplat）已在统一 eof3r env 中 E2E 验证通过。
> **定位转变**：从"拼接预训练模型"转向"跨模型几何蒸馏"——VGGT 是训练时的 teacher，MVSplat 是推理时唯一的模型。

---

## 三个机制性失败模式（根本原因，不是调参能解决的）

### 🔴 失败模式 1：Opacity ≠ Occupancy

**现象**：`opacity_mean=0.28, pass_rate(α>0.5)=2.5%`，但 BEV 中每个 α 值都没有物理含义。

**机制**：MVSplat 用 3DGS 的 alpha-blending 渲染方程优化 opacity：
```
C_pixel = Σ α_i · c_i · Π(1-α_j)
```
α 和 SH 颜色 c 是**联合优化**的——低 α + 高 c 与 高 α + 低 c 可以产生相同像素。优化器只关心渲染颜色，不关心 α 是否有物理意义。一个 α=0.28 的高斯球可以是场景表面的主要贡献者（因为它是光线首先遇到的那个），但在 BEV 中我们把它当成"28% 占据"——这是类别错误（category error）。

**根因**：MVSplat 的 opacity 是"对渲染颜色的**相对贡献权重**"，不是"该空间位置被物体占据的**概率**"。

**干预**：用 VGGT depth 做 binary silhouette 监督——该像素有深度值 → 对应光线上的高斯球应该输出 occupation=1（被占据），没有深度 → occupation=0（自由空间）。将 occupancy 从颜色渲染中解耦。

---

### 🔴 失败模式 2：协方差结构在 BEV 投影中丢失

**现象**：BEV 投影时用各向同性 `scatter + gaussian_smooth(σ=avg_scale×3)`，完全丢失 3D 协方差。

**机制**：3DGS 每个高斯球有完整 3×3 协方差 Σ（编码形状、大小、朝向）。投影到 BEV（XY 平面）时应做高度维度的**边缘化**（marginalization）：从 Σ 提取 XZ 子矩阵得到 2D 协方差，保持各向异性和朝向信息。当前 scatter+smooth：
1. 取中心 (x,z) → **丢弃 Σ**
2. 用 `avg(scale_x, scale_z)` 作为固定 σ → **忽略各向异性**
3. 各向同性 Gaussian smooth → **忽略 XZ 相关性**

**结果**：10cm 宽、1m 高的椅腿高斯球被放大为 `3×max(scale)` 的圆形——严重过度膨胀。

**干预**：可微 BEV 边缘化——对每个高斯球的 Σ 在高度维度做解析投影，保留完整 2D 协方差。等价于光线从上方穿过高斯球累积 opacity。

---

### 🔴 失败模式 3：无自由空间建模

**现象**：Costmap 中 lethal=55%, free=42%，无法区分"被占据"和"未观察到"。

**机制**：VGGT pointmap 给了表面点→我们全部当作"占据"→投影到 BEV。但每个 VGGT 像素对应一条从相机到表面的**光线**：
```
相机 ─FREE→ [沙发表面] ─UNKNOWN→ [墙（被遮挡）]
```
正确的占据模型应对每条 VGGT 光线做 free-space carving：
- 相机到表面：**FREE**
- 表面附近（±σ）：**OCCUPIED**
- 表面后方：**UNKNOWN**（被遮挡，无法确定）

**干预**：Ray-based free-space carving——用 VGGT depth 对每条光线标记三区域。这些标记作为训练监督：MVSplat 的高斯球在 free 区域应输出 occupancy=0，在 occupied 区域应输出 occupancy=1。

---

## 跨模型几何蒸馏：三个干预的统一框架

**核心创新**：VGGT 从"pipeline 的一个阶段"重新定位为"MVSplat 的几何老师"。

```
训练时：
  VGGT → depth + pointmap + free-space rays → 几何监督
  SAM2/YOLO → 2D masks → 语义监督
  MVSplat → 学习预测 occupancy + semantic + confidence（非 opacity + SH + color）

推理时：
  图像 → MVSplat → BEV occupancy + semantic costmap（单模型，前馈）
```

**三个消融实验**（论文核心）：

| 消融 | 干预 | 验证指标 |
|------|------|---------|
| A vs Baseline | 用 VGGT depth 监督 occupancy head 替代 opacity | BEV coverage、occupancy accuracy |
| B vs A | 加入可微 BEV 边缘化（保留 Σ） | Footprint IoU、boundary precision |
| C vs B | 加入 free-space carving | Costmap free/occupied/unknown 分布 |

---

## 已解决的工程问题（存档）

- ✅ SAM2 过分割 + 无语义：YOLOv8-nano (6MB) → SAM2 box-prompt，65→3 objects，真实 COCO 标签
- ✅ 坐标系不匹配：VGGT/MVSplat 坐标帧统一 + OpenCV→Y-up 转换，drivable conflict 100%→0%
- ✅ VGGT scale 归一化：地面平面恢复 ×7.834 尺度因子 + 动态 BEV grid
- ✅ Fusion 速度：130s → 0.05s（矢量化 bincount + gaussian_filter）
- ✅ eof3r 统一环境：SAM2 + VGGT + MVSplat 三个真模型均在同一 env 验证通过
- ✅ 项目解耦：wrappers 优先使用 pip 安装的包，baselines/ 仅作开发 fallback
- ✅ 项目重构：代码集中于 `eof3r/`，文档集中于 `docs/`
- ✅ SAM2/VGGT clone（GitHub TLS）：通过代理解决

---

## 环境信息（更新于 2026-06-19）

**唯一环境**：`eof3r` — Python 3.10, torch 2.5.1+cu121, GPU NVIDIA RTX A6000 (48GB)
**峰值显存**（三模型全部加载）: ~6.4 GB
**新增依赖**：ultralytics (YOLOv8)
