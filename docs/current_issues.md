# 当前问题与解决方案

> 更新时间：2026-06-19
> Phase A 消融 + Phase A.1 POC 均已完成。方向确认：**跨模型几何蒸馏**——VGGT-Ω 为 teacher，ReSplat 为 student。
> 损失函数已完成严谨数学推导（`docs/lit_notes/phaseb_design_2026-06-19.md` §2）。

---

## 三个机制性失败模式 + POC 定量证据

### 🔴 失败模式 1：Opacity ≠ Occupancy

**POC 实验定量证据**（`test_occupancy_head.py`, 2026-06-19）：
- VGGT depth 投影标记：2.6% occupied, 68.9% free, 28.5% unknown
- Post-hoc MLP val acc=96.5% 但 BEV coverage<1%（所有方法）
- **根因升级**：不仅是 opacity 值的问题——**Gaussian 位置本身就不对**（仅 2.6% 靠近 VGGT 表面，为渲染优化而非几何准确）
- **结论**：必须端到端重训 decoder + Gaussian adapter，几何 loss 反传至 μ, Σ

**机制**：MVSplat 用 3DGS 的 alpha-blending 渲染方程优化 opacity，α 与 SH 颜色联合优化。α 是渲染权重，不是占据概率。高斯位置也被渲染损失驱动——低 α 的高斯可以放在任何位置，只要不影响最终像素颜色。

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

## 跨模型几何蒸馏：Phase B 最终方案

**核心架构**（详见 `docs/lit_notes/phaseb_design_2026-06-19.md`）：

```
训练时：
  VGGT-Ω → depth + pointmap + free-space rays → 几何监督（frozen）
  SAM2/YOLO → 2D masks → 语义监督
  ReSplat → 学习预测 occupancy + semantic + free-space（非 opacity + SH + color）

推理时：
  图像 → ReSplat → BEV occupancy + semantic costmap（单模型，前馈）
```

**损失函数**（概率占据场 → NLL 推导）：

| 损失 | 公式 | 目的 |
|------|------|------|
| L_depth | Chamfer(高斯中心, VGGT表面点) | 高斯位置监督 |
| L_occ | Focal Loss(γ=2) + 类平衡 | 占据分类 |
| L_free | Hinge(o_i - 0.05)² | 自由空间正则化（独有创新） |
| L_sem | Cross-Entropy + label smoothing | 语义分类 |
| L_color | λ·(L1 + SSIM), λ=0.1 | 辅助，防encoder退化 |

**三阶段训练**：Warmup(几何)→Main(占据+自由空间+语义)→Fine-tune(颜色退场)

---

## 已解决的工程问题（存档）

- ✅ SAM2 过分割 + 无语义：YOLOv8-nano (6MB) → SAM2 box-prompt，65→3 objects，真实 COCO 标签
- ✅ 坐标系不匹配：VGGT/MVSplat 坐标帧统一 + OpenCV→Y-up 转换，drivable conflict 从无到有（证明 FG 在 BG 地面上）
- ✅ VGGT scale 归一化：地面平面恢复 ×7.834 尺度因子 + 动态 BEV grid
- ✅ Fusion 速度：130s → 0.05s（矢量化 bincount + gaussian_filter）
- ✅ eof3r 统一环境：SAM2 + VGGT + MVSplat 三个真模型均在同一 env 验证通过
- ✅ 项目解耦：wrappers 优先使用 pip 安装的包，baselines/ 仅作开发 fallback
- ✅ 项目重构：代码集中于 `eof3r/`，文档集中于 `docs/`
- ✅ SAM2/VGGT clone（GitHub TLS）：通过代理解决
- ✅ 2026-06-19 消融复验：4 变体 × 3 帧配对复现，固定 grid coverage=1.88% vs 动态 grid 85.5%（自适应假象确认）
- ✅ Conda 非交互 shell 问题：已文档化 workaround

---

## 环境信息（更新于 2026-06-19）

**唯一环境**：`eof3r` — Python 3.10, torch 2.5.1+cu121, GPU NVIDIA RTX A6000 (48GB)
**峰值显存**（三模型全部加载）: ~6.4 GB
**新增依赖**：ultralytics (YOLOv8-nano, 6MB)
**Conda**：非交互 shell 需 `source ~/anaconda3/etc/profile.d/conda.sh && conda activate eof3r`；交互终端开箱即用。

### 复验 Pipeline 命令

```bash
# E2E pipeline: fixed-grid metrics (coverage 1.88%, IoU 0.0047)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate eof3r
python eof3r/scripts/eval/test_e2e_pipeline.py

# Ablation: 4 variants × 3 frame pairs (dynamic-grid metrics)
python eof3r/scripts/eval/ablation_study.py
```
