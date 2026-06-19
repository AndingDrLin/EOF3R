# 当前问题与解决方案

> 更新时间：2026-06-19（AutoLab 8 实验完成后）
> Phase A 消融 + Phase A.1 POC 均已完成。方向确认：**跨模型几何蒸馏**——VGGT 为 teacher，ReSplat 为 student。
> 损失函数已完成严谨数学推导（`docs/lit_notes/phaseb_design_2026-06-19.md` §2）。
> AutoLab 完成 8 个 mock-data 实验，验证了训练 pipeline 和 loss 函数的通用性质。

---

## 当前阶段：Phase B — 准备真实数据训练

### ✅ 已完成

1. **训练模块完整实现**：losses, heads, supervision, trainer, train script
2. **训练 pipeline 验证**：8 个 mock-data 实验全部成功（0 NaN, 0 crash）
3. **Loss 函数消融**（mock data）：
   - Focal loss 比 BCE 好 3.5×（0.208 vs 0.723）— 类别不平衡场景下 BCE 不可用
   - 3-stage schedule 比 uniform 好 20.6%（0.359 vs 0.433）
   - 30K steps 已收敛，50K 无额外收益
4. **代码改进**：MockEncoder、eval_step 修复、loss weight CLI 参数

### 🔴 当前瓶颈（按优先级）

#### 瓶颈 1：没有真实 VGGT supervision 数据

所有 AutoLab 实验用的是 mock 数据（随机 Gaussian 参数和标签）。这意味着：
- Loss 下降不代表模型学到了几何
- Chamfer/hinge/kappa 的真实贡献被噪声掩盖
- 核心创新（occupancy head 替代 opacity）**未被验证**

**解决**：用原版 VGGT 对 Re10k 数据预计算 supervision（depth + pointmap + free-space rays）。

#### 瓶颈 2：Mock encoder 不产生真实 Gaussian 参数

当前 MockEncoder 是 `nn.Parameter(torch.randn(...))`，不看输入图像。
训练 loss 下降只说明 head 学会了分类随机标签，不代表 Gaussian 位置被拉到真实表面。

**解决**：加载真实 ReSplat encoder（预训练权重），冻结 encoder 训练 heads。

#### 瓶颈 3：ReSplat 环境隔离

ReSplat 需要 Python 3.12 + PyTorch 2.7.0 + CUDA 12.8，与 eof3r env 不兼容。

**解决**：创建独立 `resplat` conda env。VGGT supervision 预计算在 eof3r env 完成，训练在 resplat env 运行。

#### 瓶颈 4：Phase C（可微 BEV）未开始

协方差结构丢失是 Phase A 确认的第二大失败。当前 BEV 投影不可微，无法端到端训练。

**解决**：Phase C 实现解析 Σ→XZ 投影 + 可微 BEV 边缘化。

---

## 三个机制性失败模式 — 修复状态

| # | 失败模式 | 修复方案 | 当前状态 | 验证标准 |
|---|---------|---------|---------|---------|
| 1 | **Opacity ≠ Occupancy** | Occupancy head + 端到端重训 | ❌ 未验证（mock data） | 训练后 >50% Gaussians 靠近 VGGT 表面（当前 2.6%） |
| 2 | **协方差结构丢失** | 可微 BEV 边缘化（Phase C） | ⬜ 未开始 | BEV coverage 从 1.88% 提升到 >30% |
| 3 | **无自由空间建模** | VGGT 光线 free-space carving | ❌ 未验证（mock data） | Costmap lethal 从 55% 降到 <20% |

---

## 教师模型选型决策（2026-06-19 更新）

**原计划**：VGGT-Ω (CVPR 2026 Oral) 作为 geometry teacher。
**当前决策**：先用**原版 VGGT**，最后阶段再切换到 VGGT-Ω。

**原因**：
1. VGGT-Ω checkpoint 需要 HuggingFace gated 访问权限，申请流程不确定
2. VGGT-Ω 需要 Python 3.12 + PyTorch 2.7，环境搭建复杂
3. 原版 VGGT 已验证可用（13.6s, 1B model, depth δ1.25=67.5%）
4. 先用 VGGT 跑通整个 pipeline，验证核心方法有效，再升级 teacher 不影响结论
5. VGGT-Ω 的优势（+26% depth 精度, 1.6× 更快）是锦上添花，不是方法论依赖

**切换时机**：Phase B/C/D 全部完成、论文实验跑完后，用 VGGT-Ω 替换 VGGT 重新跑一遍，作为最终结果。

---

## 已解决的工程问题（存档）

- ✅ SAM2 过分割 + 无语义：YOLOv8-nano (6MB) → SAM2 box-prompt，65→3 objects，真实 COCO 标签
- ✅ 坐标系不匹配：VGGT/MVSplat 坐标帧统一 + OpenCV→Y-up 转换，drivable conflict 从无到有
- ✅ VGGT scale 归一化：地面平面恢复 ×7.834 尺度因子 + 动态 BEV grid
- ✅ Fusion 速度：130s → 0.05s（矢量化 bincount + gaussian_filter）
- ✅ eof3r 统一环境：SAM2 + VGGT + MVSplat 三个真模型均在同一 env 验证通过
- ✅ 2026-06-19 消融复验：4 变体 × 3 帧配对复现，三个失败模式确认
- ✅ 2026-06-19 AutoLab：8 实验完成，训练 pipeline 验证，focal loss + stage schedule 确认有效
- ✅ 2026-06-19 Trainer 修复：eval_step (no_grad backward bug) + MockEncoder + CLI args

---

## 环境信息（更新于 2026-06-19）

**主环境**：`eof3r` — Python 3.10, torch 2.5.1+cu121, GPU NVIDIA RTX A6000 (48GB)
**峰值显存**（三模型全部加载）: ~6.4 GB
**Conda 路径**：`/home/ubuntu/lyj/anaconda3/`（非交互 shell 需 source + activate）
**GPU 显存安全阈值**：<90%（44GB）

### 复验 Pipeline 命令

```bash
# E2E pipeline: fixed-grid metrics (coverage 1.88%, IoU 0.0047)
source /home/ubuntu/lyj/anaconda3/etc/profile.d/conda.sh && conda activate eof3r
python eof3r/scripts/eval/test_e2e_pipeline.py

# Ablation: 4 variants × 3 frame pairs (dynamic-grid metrics)
python eof3r/scripts/eval/ablation_study.py

# AutoLab: Phase B training with mock data
python eof3r/scripts/train/train_phase_b.py --batch-size 4 --total-steps 30000 --device cuda
```
