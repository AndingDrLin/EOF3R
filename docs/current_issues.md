# 当前问题与解决方案

> 更新时间：2025-06-18
> 所有真模型（SAM2+VGGT+MVSplat）已在统一 eof3r env 中 E2E 验证通过。
> 以下为 BEV 占据质量问题（上次 E2E 运行发现）。

---

## 🔴 问题 1：坐标系不匹配

**证据**：
- `bev_occupancy_coverage_t0.3 = 0.0045`（仅 0.45% 的 BEV cell 被占据）
- `bev_spatial_extent_m2 = 2.24`（对于一个房间极度不合理）
- MVSplat Gaussian 空间范围 X:[-36.6, 0.2], Z:[0.55, 86]，BEV 网格只有 20m×20m
- FG/BG overlap IoU = 0.22（FG 高斯球与 BG pointmap 几乎没有重叠）
- Drivable conflict rate = 100%（FG 说被占的地方，BG 全说 drivable）

**根因**：MVSplat 输出的高斯球坐标系统与 VGGT 输出的 pointmap 坐标系统不一致，且两者都未与项目的 Y-up 约定对齐。

**修复方向**：
1. 分别检查 MVSplat 编码器输出的坐标系（可能是相机坐标系或 MVSplat 内部归一化坐标）
2. 检查 VGGT `world_points` 的实际坐标系（以首帧相机为原点？）
3. 寻找两个模型输出间的坐标变换（scale/rotation/translation）
4. 在 fusion 阶段加入坐标系校准参数（`configs/fusion.yaml`）
5. 增大 BEV grid range 或根据实际场景动态调整

---

## 🔴 问题 2：BEV 网格覆盖不足

**证据**：高斯球空间范围 36m×36m，BEV grid 仅 20m×20m。90% 以上的高斯球在 grid 边界外被裁剪。

**根因**：`configs/default.yaml` 的 `fusion.bev_range=[-10, -10, 10, 10]` 是固定值，不匹配 Re10k 场景的尺度。

**修复方向**：
1. 根据实际高斯球的 spatial range 自动计算 BEV range
2. 或增大默认 BEV range 到 ±40m
3. 对高斯球坐标做 scale 归一化后再投影

---

## 🟡 问题 3：SAM2 过分割

**证据**：在单张 720×1280 室内图像上检测到 65 个 object，标签全是占位符轮询。

**根因**：(a) SAM2 automatic mode 的 `points_per_side=32` 对纹理丰富的场景过分割，(b) 没有语义分类器。

**修复方向**：
1. 调高 `pred_iou_thresh` 和 `stability_score_thresh`，减少碎片
2. 集成为 YOLO-World 做框检测 → SAM2 做精细 mask（lit review 推荐的方案）
3. 使用 `box_prompt=True` 模式代替 automatic mode

---

## 🟡 问题 4：矢量化融合导致峰值稀释

**证据**：BEV coverage 从老算法的 67%（合成数据）降到 0.45%（真数据）。部分原因是问题 1+2（grid 太小），但也因为 scatter+gaussian_smooth 的归一化策略不对。

**根因**：`_gaussian_smooth` 后除以 `bev_max` 做 re-normalization，但 grid 内只有少数点，绝大多数 cell 的 occupancy 被稀释到 0.3 以下。

**修复方向**：解决问题 1+2 后重新评估；如果仍然过低，改用 `np.maximum.at` 实现 max-mode scatter（更接近老算法的行为）。

---

## 🟡 问题 5：无语义分类器

**证据**：SAM2 的 labels 全是 `["unknown", "person", "bicycle", "cone", ...]` 轮询，与图像内容无关。

**根因**：SAM2 不提供语义标签。当前 wrapper 硬编码了一个占位符列表。

**修复方向**：
1. 在 SAM2 输出的 bbox 上运行轻量分类器（如 CLIP zero-shot 或 YOLO 检测结果）
2. 将 semantic label 写入 Gaussian metadata
3. Costmap generator 已有 `semantic_weights`（person=1.5, cone=0.8...），只等真实标签

---

## 已解决的问题（存档）

- ✅ Fusion 速度：130s → 0.05s（矢量化 bincount + gaussian_filter）
- ✅ eof3r 统一环境：SAM2 + VGGT + MVSplat 三个真模型均在同一 env 验证通过
- ✅ MVSplat torch 兼容：通过 sys.path + sys.modules 隔离解决 src/ 命名冲突
- ✅ 公开数据集：Re10k 4 帧 720p 图像已保存到 `data/public/re10k_samples/`
- ✅ SAM2/VGGT clone（GitHub TLS）：通过代理 192.168.213.103:53941 解决
- ✅ 项目重构：代码集中于 `eof3r/`，文档集中于 `docs/`

---

## 环境信息（更新于 2025-06-18）

**唯一环境**：`eof3r` — Python 3.10, torch 2.5.1+cu121, GPU NVIDIA RTX A6000 (48GB)
**峰值显存**（三模型全部加载）: ~6.3 GB
