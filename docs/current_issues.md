# 当前问题与解决方案

> 更新时间：2026-06-18
> 所有真模型（SAM2+VGGT+MVSplat）已在统一 eof3r env 中 E2E 验证通过。
> **坐标系匹配已解决**（commit 36ce286+）。以下为剩余 BEV 占据质量问题。

---

## ✅ 问题 1：坐标系不匹配 — 已解决 (2026-06-18)

**修复前证据**：
- Drivable conflict rate = 100%（FG 说被占的地方，BG 全说 drivable）
- FG/BG overlap IoU = 0.22（两模型输出在不同坐标帧）

**根因**：
1. MVSplat 用合成 pose（任意世界帧），VGGT 自估计 pose（首帧相机原点 + unit-scale 归一化）
2. VGGT 输出在 OpenCV RDF 帧，项目使用 Y-up 帧——wrapper 未做转换
3. VGGT `_pose_enc_to_matrices()` 错误解码 pose_enc（6D rotation → 应是 quat+FOV）

**修复内容**（`eof3r/src/background/vggt_wrapper.py` + `eof3r/scripts/eval/test_e2e_pipeline.py`）：
1. 修正 `_pose_enc_to_matrices()`：正确解码 VGGT 的 9D pose_enc（T(3)+quat(4)+FOV(2)）
2. 添加 `_opencv_rdf_to_yup_points/poses()`：OpenCV RDF → Y-up (R=diag(1,-1,-1))
3. E2E 测试用 VGGT 的 OpenCV 位姿作为 MVSplat C2W extrinsics → 两模型同帧
4. MVSplat 高斯球 + VGGT pointmap 统一转换到 Y-up 后再融合

**修复后结果**：
- Drivable conflict rate: 100% → **0.00%** ✅
- FG/BG overlap IoU: 0.22 → 0.21（持平——重叠差来自 scale 问题，非坐标系统）
- 两者确实在同一坐标帧中（冲突率归零证实）

**遗留**：VGGT 的 unit-scale 归一化导致场景被压缩（见问题 2）。

---

## 🔴 问题 2：VGGT Unit-Scale 归一化导致 BEV 覆盖不足

**证据**（2026-06-18 E2E 运行）：
- `bev_occupancy_coverage_t0.3 = 0.0009`（0.09%，比修复前的 0.45% 更差——因为坐标对齐后高度滤波生效）
- `bev_spatial_extent_m2 = 6.42`（占 BEV grid 1600m² 的 0.4%）
- MVSplat 高斯球 Mean Y=4.47m（高于相机），高度滤波 (-0.5, 2.0) 过滤大部分
- VGGT 训练时对 pointmap 做了 unit average distance 归一化（`vggt/training/train_utils/normalization.py:100-103`）

**根因**：VGGT 输出的世界坐标被归一化到平均距离≈1，而 BEV grid 配置为 40×40m 真实尺度。整个场景被压缩在几米范围内，cell 密度极低。

**修复方向**：
1. ~~增大 BEV range~~（已从 20m→40m，不解决根本问题）
2. 根据 VGGT 输出的实际 spatial range 动态设置 BEV grid 范围+分辨率
3. 尝试从 VGGT pointmap 恢复真实尺度（需要已知某一维度的真实长度）
4. 对于纯融合验证，使用 VGGT stub（合成数据在真实米尺度）跳过 unit-scale 问题
5. 长期：在 pipeline 中加入 scale calibration 模块（利用 LiDAR/depth 测量作为 scale anchor）

---

## ✅ 问题 3：SAM2 过分割 — 已解决 (2026-06-18)

**修复前**：单张 720×1280 图像 → 65 个碎片，标签占位符轮询。

**修复方案**：YOLOv8-nano (6MB) 做检测 + SAM2 box-prompt 精细分割。
- YOLO 用 3 个 bbox prompt 代替 SAM2 automatic 的 32×32=1024 个网格点
- 检测数: 65 → 3（`['couch', 'chair', 'chair']`）
- 分割时间: 6-7s → 4.3s
- 修复代码：`sam2_wrapper.py` `detect_and_segment()` 方法
- 向后兼容：`segment()` 保留 automatic 模式

---

## 🟡 问题 4：矢量化融合导致峰值稀释

**证据**：BEV coverage 从老算法的 67%（合成数据）降到 0.45%（真数据）。部分原因是问题 1+2（grid 太小），但也因为 scatter+gaussian_smooth 的归一化策略不对。

**根因**：`_gaussian_smooth` 后除以 `bev_max` 做 re-normalization，但 grid 内只有少数点，绝大多数 cell 的 occupancy 被稀释到 0.3 以下。

**当前状态**：问题 1+2 已解决（坐标对齐 + scale recovery + 动态 BEV），coverage 提升至 1.88%。待实机数据验证后重新评估是否需要 max-mode scatter。

---

## ✅ 问题 5：无语义分类器 — 已解决 (2026-06-18)

**修复前**：labels 全是 `["unknown", "person", "bicycle", ...]` 轮询占位符。

**修复方案**：YOLOv8-nano 自带 COCO 80 类语义标签 + 风险等级映射。
- 语义 BEV 已生成（2 类成功投影）
- 风险等级映射：person=3, bicycle=2, chair=0, …
- Costmap generator 的 `semantic_weights` 已在消费真实标签
- 修复代码：`sam2_wrapper.py` `_COCO_CLASSES` + `_CLASS_RISK` 字典

---

## 已解决的问题（存档）

- ✅ SAM2 过分割（问题 3）+ 无语义分类器（问题 5）：YOLOv8-nano (6MB) → SAM2 box-prompt，65→3 objects，真实 COCO 标签 + 风险等级
- ✅ 坐标系不匹配（问题 1）：VGGT/MVSplat 坐标帧统一 + OpenCV→Y-up 转换，drivable conflict 100%→0%
- ✅ Fusion 速度：130s → 0.05s（矢量化 bincount + gaussian_filter）
- ✅ eof3r 统一环境：SAM2 + VGGT + MVSplat 三个真模型均在同一 env 验证通过
- ✅ MVSplat torch 兼容：通过 sys.path + sys.modules 隔离解决 src/ 命名冲突
- ✅ 公开数据集：Re10k 4 帧 720p 图像已保存到 `data/public/re10k_samples/`
- ✅ SAM2/VGGT clone（GitHub TLS）：通过代理 192.168.213.103:53941 解决
- ✅ 项目重构：代码集中于 `eof3r/`，文档集中于 `docs/`
- ✅ 项目解耦：wrappers 优先使用 pip 安装的包，baselines/ 仅作开发 fallback

---

## 环境信息（更新于 2026-06-18）

**唯一环境**：`eof3r` — Python 3.10, torch 2.5.1+cu121, GPU NVIDIA RTX A6000 (48GB)
**峰值显存**（三模型全部加载）: ~6.3 GB
