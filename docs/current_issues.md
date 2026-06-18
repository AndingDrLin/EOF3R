# 当前问题与解决方案

> 更新时间：2025-06-18
> 记录真模型 E2E 验证后发现的问题及解决方案。

---

## 问题 1：Fusion BEV 投影速度（P0）

**现象**：VGGT-1B 输出约 190k 个 3D 点，`bev_projector.project_gaussians_to_bev()` 使用 Python for 循环逐点投影到 400×400 BEV 网格，耗时 130s。

**根因**：`bev_projector.py` 第 106-125 行的逐高斯球遍历 + 逐 cell 写入是纯 Python `for i in range(N)`。

**解决方案**：

方案 A（推荐）— **numpy 矢量化 scatter**：
```python
# 用 np.histogram2d 替代逐点遍历
cols = ((mx - x_min) / resolution).astype(np.int32)
rows = ((my - y_min) / resolution).astype(np.int32)
# 过滤 + 2D 直方图加权
valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
bev = np.histogram2d(rows[valid], cols[valid], bins=(h, w),
                      weights=opacities[valid])[0]
```

方案 B — **CUDA kernel**（如方案 A 不够快）：
用 `torch.scatter_add` 或写一个简单的 CUDA 核函数，所有点并行写入 BEV grid，利用 atomic operations。

方案 C — **稀疏采样**：
VGGT 的 190k 点中有大量冗余（逐像素密集点云）。先用 `np.random.choice` 采样到 10k 点再投影，精度损失 <5%。

**预期**：方案 A → <1s，方案 B → <100ms。

**实施状态**：待实施

---

## 问题 2：MVSplat 真模型推理（P1）

**现象**：MVSplat（torch 2.1 + hydra 特定版本）与 eof3r 主环境（torch 2.5.1）不兼容，无法在同一 Python 进程中同时 import MVSplat internals 和 SAM2/VGGT。

**根因**：
- MVSplat 依赖 `hydra` + `src.config.load_typed_root_config`（mvsplat 内部模块）
- mvsplat env 的 torch 2.1.2 与 SAM2 要求的 torch>=2.5.1 冲突
- eof3r env 的 torch 2.5.1 与 MVSplat 的 `torch.cuda.amp` API 变更不兼容

**解决方案**：

方案 A（推荐）— **subprocess 隔离**：
```python
# foreground 阶段通过 subprocess 调用 mvsplat env
subprocess.run([
    "conda", "run", "-n", "mvsplat", "python",
    "scripts/foreground_infer.py",
    "--input", images_path,
    "--output", gaussians_path,
])
# 主进程加载 .npy 文件继续 pipeline
```

方案 B — **Docker/独立服务**：将 MVSplat 作为独立 HTTP 服务运行，与其他模块解耦。

**实施状态**：待实施。短期 workaround: `--skip-mvsplat` 使用合成高斯球。

---

## 问题 3：真数据验证（P1）

**现象**：E2E 测试中 SAM2 + VGGT 用合成随机图像，FG 用合成高斯球。FG/BG overlap IoU = 0.005（完全不匹配），无法评估真实融合质量。

**根因**：Re10k 测试图像加载依赖 MVSplat 内部 dataloader（`src.dataset.data_module`），该模块在 eof3r env 中不可用。

**解决方案**：

方案 A — **直接保存 Re10k 样本帧**：
在 mvsplat env 中运行一次 MVSplat dataloader，保存 2-4 帧 RGB 图像到 `data/test_fixtures/`，eof3r env 直接加载。

方案 B — **使用 COCO/ADE20K 公开图像**：
从 COCO 等公开数据集中选取几张含物体的室内/室外图像，保存为 test fixtures。

**实施状态**：待实施

---

## 问题 4：VGGT 地面/可通行估计未验证（P2）

**现象**：VGGT 输出的 drivable_mask 覆盖率 0.36%，地面平面参数 [0, 1, 0, 0]（水平平面），但未与真实场景对比验证。

**根因**：无 GT 数据（无 campus rosbag、无 ScanNet++）。

**解决方案**：待 Stage 2 数据准备完成后验证。短期用可视化人工检查 VGGT pointmap 质量。

**实施状态**：待 Stage 2

---

## 环境信息（更新于 2025-06-18）

| 环境 | Python | Torch | CUDA | 包含模型 |
|------|--------|-------|------|----------|
| `eof3r` | 3.10 | 2.5.1+cu121 | 12.1 | SAM2 + VGGT-1B |
| `mvsplat` | 3.10 | 2.1.2+cu118 | 11.8 | MVSplat + VGGT-1B |

**GPU**: NVIDIA RTX A6000 (48GB VRAM)
**峰值显存**（VGGT-1B 推理）: ~5.8 GB
