# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## §1 项目身份与语言

**EOF3R** — Efficient Object-level Feedforward 3D Reconstruction with 3DGS.

**核心创新**：将 feedforward 3DGS 从 photorealistic 渲染工具改造为 planning-oriented 几何-语义占据预测器。不是"拼接预训练模型"，而是**跨模型几何蒸馏（cross-model geometric distillation）**——用一个前馈几何模型（VGGT）的输出作为另一个前馈渲染模型（MVSplat）的训练监督信号，使后者学会预测 metric-scale 占据而非逼真颜色。

**Despite the name, the project produces planning-oriented Gaussian occupancy, not photorealistic images.**

本科毕业设计原型系统。面向 Husky 低速无人车在校园/园区的"最后 50 米"配送场景。

**方法本质**：
- **训练时**：VGGT 提供 depth、free-space ray、pointmap 作为几何监督 → MVSplat 学习用 Gaussian primitives 预测 occupancy + semantic + confidence（而非 opacity + SH + color）
- **推理时**：只需要 MVSplat（单模型，前馈）→ 直接从 RGB 输出 metric-scale BEV 占据 + 语义 costmap
- VGGT 是"几何老师"，不是"pipeline 阶段"——这是与"拼接方案"的本质区别

**系统架构**：车端始终运行本地安全回路（相机/里程计/IMU/急停/Nav2 局部规划/cmd_vel），云端运行改造后的 MVSplat（单模型前馈推理）。SAM2/YOLO 提供 2D mask 监督（训练时），VGGT 提供几何监督（训练时）。推理时只跑 MVSplat + 轻量分割，输出 BEV occupancy + semantic costmap。

**系统不追求逼真重建，而追求更可靠、更适合规划的几何-语义表示。**

### 语言铁律

**只有两种东西用中文：**
1. 与用户对话
2. 项目文档文件（`docs/`、`lit_notes/`、`experiments/`、`README.md` 等面向人阅读的 markdown 文件）

**其余一切用英文：**
- Code comments, commit messages, variable names, function names, CLI arguments, log output, error messages, config field names, filenames

**模型名、方法名、学术术语保留原英文，不翻译。**

This file itself is in Chinese for §1 and English for code-facing sections — reflecting the two-zone rule.

---

## §1b Development Commands

```bash
# End-to-end pipeline test (SAM2 + VGGT + MVSplat, all real models)
conda activate eof3r
python eof3r/scripts/eval/test_e2e_pipeline.py
# Outputs: outputs/eval/e2e_metrics.json + e2e_pipeline_visualization.png

# BEV projection verification (MVSplat only, auto-resolves MVSplat root)
conda activate eof3r
python eof3r/scripts/eval/verify_mvsplat_bev.py

# Lint & format (auto-fix)
ruff check --fix eof3r/src/ eof3r/scripts/ --config eof3r/pyproject.toml
ruff format eof3r/src/ eof3r/scripts/ --config eof3r/pyproject.toml

# Setup a baseline
bash eof3r/scripts/setup_mvsplat.sh
```

### Environment

```bash
# Single unified environment (2026-06-18)
conda activate eof3r  # Python 3.10 + torch 2.5.1 + CUDA 12.1 + all models

# To recreate from scratch:
conda create -n eof3r python=3.10 -y && conda activate eof3r
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/sam2.git
pip install git+https://github.com/facebookresearch/vggt.git
# For MVSplat: git clone https://github.com/donydchen/mvsplat && export MVSPLAT_ROOT=/path/to/mvsplat
pip install -r eof3r/requirements.txt
# For full deps list, inspect the install commands in memory or docs/current_issues.md.

# GPU: NVIDIA RTX A6000 (48GB), peak VRAM ~6.3GB (VGGT + MVSplat together)
```

### Known Blockers (as of 2026-06-19)

- ~~环境搭建、模型安装、数据准备~~ → **全部 Resolved。**
- **当前核心 Blocker**：MVSplat 输出的是 photorealistic Gaussian primitives（opacity 与颜色纠缠、scale 无物理约束、缺失自由空间建模）→ BEV 投影不可用。解决方向：用 VGGT 的几何信号重新训练 MVSplat 的 decoder head（见 §1c 和 `docs/current_issues.md`）。
- **次要 Blocker**：无校园 rosbag。公开数据集 Re10k 作为验证替代。

### Config-driven Experiments

All experiments are driven by YAML configs inheriting from `eof3r/configs/default.yaml`. Override fields via CLI or experiment-specific YAML. Never hardcode hyperparameters in code.

---

## §1c Architecture: Cross-Model Geometric Distillation

### The Fundamental Insight

Prior work concatenates feedforward models as sequential inference stages (SAM2→VGGT→MVSplat→fusion).  This suffers from three mechanistic failures when projecting Gaussian primitives to BEV:

1. **Opacity-Occupancy Mismatch**: MVSplat's opacity is optimised for alpha-blending with colour — low α + high SH can produce the same pixel as high α + low SH.  Opacity is a *rendering weight*, not an *occupation probability*.  Treating it as occupancy is a category error.

2. **Covariance Information Loss**: Scatter+smooth BEV projection discards the full 3×3 covariance Σ of each Gaussian.  An anisotropic Gaussian (e.g., 10cm-wide chair leg, 1m tall) gets isotropically inflated to `3·max(scale)` in BEV.

3. **Missing Free-Space Modelling**: VGGT pointmap gives surface points, but ray-based free-space carving (camera→surface = FREE, surface vicinity = OCCUPIED, behind surface = UNKNOWN) is never performed.  The costmap cannot distinguish free from unknown.

These failures are NOT fixable by tuning — they stem from a category mismatch: **photorealistic primitives ≠ planning-oriented primitives**.

### The Approach: VGGT as Teacher, MVSplat as Student

```
                     TRAINING                          │            INFERENCE
                                                       │
  ┌──────┐    ┌──────┐                                 │   ┌──────┐
  │ SAM2 │    │ VGGT │  ← both provide SUPERVISION     │   │ RGB  │
  └──┬───┘    └──┬───┘                                 │   └──┬───┘
     │2D masks   │depth, pointmap, free-space rays     │      │
     │           │                                      │      ▼
     ▼           ▼                                      │   ┌──────────┐
  ┌──────────────────────────────┐                      │   │  MVSplat │
  │         MVSplat              │  ← train decoder     │   │ (infer)  │
  │  freeze: encoder (cost vol) │    heads with         │   └───┬──────┘
  │  retrain: occupancy head    │    geometric loss     │       │
  │           semantic head     │                       │       ▼
  │           confidence head   │                       │   ┌──────────┐
  └──────────────────────────────┘                      │   │   BEV    │
                                                        │   │occupancy │
  L_total = L_depth + L_occ + L_free + L_semantic       │   │+semantic │
            + λ·L_color  (λ=0.1, auxiliary only)        │   │+costmap  │
                                                        │   └──────────┘
```

**VGGT is a training-time geometry teacher, not an inference-time pipeline stage.**  At inference, only MVSplat runs — a single feedforward model directly predicting planning-oriented occupancy from RGB.

### Three Principled Interventions

| # | Failure Mode | Intervention | Supervision |
|---|-------------|-------------|-------------|
| 1 | Opacity ≠ Occupancy | Replace opacity head with **occupancy head** (sigmoid output, 0=free, 1=occupied) | VGGT depth → silhouette loss (binary: is there a surface at this depth?) |
| 2 | Covariance discarded | **Differentiable BEV marginalization** — analytically project Σ to XZ plane, preserving anisotropy | VGGT pointmap density → constrain scale to physically plausible range |
| 3 | No free-space model | **Ray-based carving** — for each VGGT ray, label space BEFORE surface as FREE, AT surface as OCCUPIED, BEHIND as UNKNOWN | VGGT depth rays + pointmap |

### Current Code Status (as of 2026-06-19)

The `eof3r/src/` modules currently implement the **sequential concatenation baseline** (for ablation comparison):

| Module | File | Purpose (Baseline) | Purpose (Target) |
|--------|------|--------------------|--------------------|
| segmentation | `sam2_wrapper.py` | Inference-stage segmentation → masks | Training supervision: 2D masks for semantic loss |
| foreground | `mvsplat_wrapper.py` | Inference-stage feedforward Gaussians | **Unified inference model** (occupancy+semantic+confidence) |
| background | `vggt_wrapper.py` | Inference-stage geometry estimation | **Training supervision**: depth, pointmap, free-space rays |
| fusion | `bev_projector.py`, `coord_utils.py` | numpy BEV projection | → Replace with differentiable BEV marginalization |
| costmap | `costmap_generator.py` | Post-processing costmap | Train-time planning loss |
| communication | `__init__.py` only | 🔴 Empty stub | Vehicle↔cloud bridge |

### Implementation Roadmap

- **Phase A** (current): Sequential baseline — all three models as inference stages.  Used for ablation: quantify how bad BEV is without geometric distillation.
- **Phase B** (next): MVSplat decoder retraining — add occupancy/semantic/confidence heads, freeze encoder, train with VGGT geometric supervision.
- **Phase C**: Differentiable BEV marginalization + ray-based free-space carving — replace numpy projection with torch operations.
- **Phase D**: End-to-end training with planning loss — backpropagate from costmap quality metrics to Gaussian parameters.

---

## §2 Directory Structure & Naming

```
EOF3R/                              # Repo root
├── README.md
├── CLAUDE.md                       # Project constitution + memory (§13)
├── .gitignore
│
├── docs/                           # ALL documentation (中文)
│   ├── project_scope.md
│   ├── project_audit.md
│   ├── roadmap.md
│   ├── lit_review.md
│   ├── experiments.md
│   ├── engineering.md
│   ├── risks.md
│   ├── standards.md
│   ├── todo.md
│   ├── current_issues.md           # Active issues with root causes & solutions
│   ├── lit_notes/                  # Paper reading notes (24 papers)
│   ├── experiments/                # Experiment logbook
│   └── thesis/                     # Paper-writing materials
│       ├── figures/ tables/ references/ notes/
│       └── writing/  (gitignored)
│
├── eof3r/                          # ALL runnable code
│   ├── src/                        # Importable Python package (8 modules)
│   ├── scripts/                    # Standalone scripts (eval/, robot/, etc.)
│   ├── configs/                    # YAML configs (default.yaml, plot_style.yaml, ...)
│   ├── tests/                      # Smoke tests per module
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .pre-commit-config.yaml
│
├── baselines/                      # External code (gitignored except registry+patches)
├── data/                           # Datasets (gitignored except test_fixtures/)
└── outputs/                        # Experiment outputs (gitignored)
```

### Naming Conventions

- **Python files**: `snake_case.py` (e.g., `gaussian_occupancy.py`)
- **Config files**: `{purpose}.yaml` (e.g., `scannet.yaml`)
- **Checkpoints**: `{model}_{dataset}_{step}.pth` (e.g., `vggt_scannet_5000.pth`)
- **Experiments**: `{stage}_{YYYYMMDD}_{short_desc}` (e.g., `s3_20250701_chair_3dgs`)
- **Experiment logs**: `experiments/{YYYY-MM-DD}_{short_desc}.md`
- **Paper notes**: `{model_keyword}_{author_year}.md` (e.g., `vggt_wang2025.md`)
- **Max directory depth**: 3 levels from repo root

---

## §3 Baseline Management

External open-source code (3DGS, VGGT, SAM2, DUSt3R) lives under `baselines/`.

### Rules

1. **Always use `baselines/registry.yaml`** — before cloning any baseline, add an entry (even commented out) recording URL, branch, commit hash, clone date, and conda environment name. Update commit hash after clone.

2. **Never modify baseline source code directly.** If a change is needed:
   - Create a `.patch` file in `baselines/patches/{name}.patch`
   - Document in registry.yaml `patched: true`
   - Apply via `git apply` in setup scripts

3. **Wrap every baseline behind a Python interface** in the corresponding `eof3r/src/` module. Example:
   - `eof3r/src/foreground/mvsplat_wrapper.py` wraps `baselines/mvsplat/`
   - `eof3r/src/background/vggt_wrapper.py` wraps `baselines/vggt/`
   - All wrappers expose a consistent API (`build()`, `train()`, `infer()`, `save()`, `load()`)

4. **Isolate environments** — each baseline may have its own conda env (recorded in registry.yaml). Provide a setup script in `eof3r/scripts/setup_{baseline}.sh`.

5. **Pre-trained weights** go to `outputs/checkpoints/{baseline_name}/`, not inside `baselines/`.

6. **Baseline code is gitignored.** Only `registry.yaml` and `patches/` are committed.

### Current Status (2026-06-18)

| Baseline | Status | Notes |
|----------|--------|-------|
| MVSplat | 🟢 Cloned + checkpoints | re10k.ckpt, acid.ckpt |
| DepthSplat | 🟢 Cloned | Not yet used |
| SAM2 | 🟢 Cloned + verified | HuggingFace auto-download, 65-object over-segmentation on Re10k (needs YOLO preprocessing) |
| VGGT | 🟢 Cloned + verified | HuggingFace auto-download, 1B model, ~14s inference |
| DUSt3R, MASt3R | ⬜ Not started | — |
| Nav2 | ⬜ Not started | Installed via apt on Husky only |

When cloning is blocked, create a stub class (e.g., `SAM2Stub`) with the same API as the planned wrapper, generating synthetic data so downstream modules can be tested. Document the planned real API in the stub's docstring.

---

## §4 Dataset Management

### Data Root

All datasets live outside the repo at a path specified by the environment variable `EOF3R_DATA` (e.g., `/data/EOF3R/`). The repo-internal `data/` directory is gitignored and may contain symlinks.

```
$EOF3R_DATA/
├── raw/               # Original downloads (read-only — never modify)
├── processed/         # Preprocessed data (regeneratable)
└── registry.yaml      # Dataset manifest
```

### Dataset Registry (`$EOF3R_DATA/registry.yaml`)

Every dataset must be registered with: name, version, source URL, download date, license, preprocessing script path, and a brief note on what it's used for.

### Dataset Selection

- **Primary (geometry validation)**: ScanNet++ (real indoor, camera poses included)
- **Primary (navigation)**: Campus rosbag (Husky self-collected, at least 3 scenarios)
- **Backup**: Replica (synthetic, clean)
- **Simulation**: Gazebo / Isaac Sim (for system integration testing)
- **Custom**: Self-captured data + COLMAP calibration

### Rules

- Switch datasets via `eof3r/configs/default.yaml` → `data.dataset`, never hardcode dataset names in code.
- Preprocessing scripts live in `eof3r/scripts/preprocess/{dataset_name}.py`.
- Small test fixtures (<1MB) may be committed to `data/test_fixtures/`.
- Large files are never committed. Provide download URLs / scripts instead.

---

## §5 Experiment Management

### Experiment Naming

`{stage}_{YYYYMMDD}_{short_desc}`, e.g., `s3_20250701_chair_3dgs_baseline`

### Output Directory

```
outputs/results/{exp_name}/
├── config.yaml        # Full copy of the experiment config
├── metrics.json       # Quantitative results (dict: metric → value)
├── visualizations/    # Rendered views, comparison images, loss curves
├── checkpoints/       # Best model weights only (not every epoch)
└── summary.md         # (optional) human-readable summary
```

### Logging

- Use Python `logging` module. Output to both stdout and `outputs/logs/{exp_name}.log`.
- Log: timestamps, GPU memory, loss values, key metrics, hardware info.

### Experiment Logbook

Record every experiment in `experiments/{YYYY-MM-DD}_{short_desc}.md` using the template `experiments/exp_template.md`. Include: purpose, config, results, observations, failure analysis, next steps.

### Immutability

**Once an experiment is done, never modify its output directory.** To re-run, create a new experiment name.

### Fair Comparison

All methods in the same comparison use the same random seed and data split.

---

## §6 Code Standards

### Language & Tooling

- **Python**: ≥3.10
- **Formatter/Linter**: `ruff` (configured in `pyproject.toml`), line length 100
- **Pre-commit**: `ruff --fix` + `ruff-format` + basic checks (configured in `.pre-commit-config.yaml`)

### Code Quality

- **Type hints**: required on all public functions. Use `mypy`-compatible syntax.
- **Docstrings**: concise one-liner for simple functions; Google-style for complex ones. English.
- **Function length**: ≤50 lines (research code: ≤100 lines tolerated).
- **File length**: ≤500 lines (research code: ≤800 lines tolerated).
- **Import order**: stdlib → third-party → internal. Separate groups with one blank line.

### Script vs Library

- `eof3r/scripts/` — standalone runnable scripts. Must have `if __name__ == "__main__":` guard.
- `eof3r/src/` — importable modules. No side effects on import.

### Testing

- Smoke tests in `eof3r/tests/` — verify each module's pipeline can run end-to-end on a small fixture.
- Naming: `eof3r/tests/test_{module}.py`
- Not aiming for 100% coverage. Aim for "every stage can be verified with one command."

### Error Handling

- **Validate only at system boundaries**: user input, file I/O, network requests.
- Internal functions trust their callers. No defensive `assert` chains.

---

## §7 3D Computer Vision Conventions

### Coordinate System

- **3D World / Occupancy**: **Right-handed, Y-up**, unit: **meters**. OpenGL convention.
- **BEV / Robot**: **X-forward, Y-left, Z-up**, unit: **meters**. ROS convention.
- The conversion from Y-up (reconstruction) to Z-up (BEV) happens at the fusion→costmap boundary.
- If a baseline uses a different convention, document and convert at the wrapper boundary.

### Camera Model

- **Pin-hole model**, OpenCV convention: **RDF (Right-Down-Forward)**.
- Camera matrix `K` follows standard form: `[fx, 0, cx; 0, fy, cy; 0, 0, 1]`.
- If a baseline uses OpenGL convention (RUB: Right-Up-Backward), document explicitly in the wrapper.

### Rotation

- Store as **quaternion (w, x, y, z)**.
- Convert to rotation matrix for computation.

### Depth

- **Z-buffer depth** (not Euclidean distance), unit: meters.

### Image Convention

- **RGB** (not BGR), floating-point [0, 1] or uint8 [0, 255].
- All modules must be consistent. Convert at the input/output boundary if needed.

### 3DGS Format

- Follow the standard `.ply` format: `(x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3)`.
- SH coefficients (f_dc, f_rest) retained for baseline compatibility but may be discarded in the costmap pipeline — only geometry fields (x, y, z, scale, rotation, opacity) are used for occupancy. Add `occupancy_alpha` as an optional field for planning-oriented Gaussian primitives.

### Evaluation Metrics

**Primary — Planning & Occupancy Quality:**

| Domain | Metric | Direction |
|--------|--------|-----------|
| Occupancy Accuracy | Footprint IoU | ↑ |
| Occupancy Accuracy | Chamfer Distance (L1) | ↓ |
| Occupancy Accuracy | F-Score @1% / @5cm | ↑ |
| Navigation Quality | Path Smoothness (rad/m) | ↓ |
| Navigation Quality | Unnecessary Stop Count | ↓ |
| Navigation Quality | Time to Goal (s) | ↓ |
| Navigation Quality | Minimum Clearance (m) | ↑ |
| Perception Latency | Per-frame inference time (ms) | ↓ |
| Perception Latency | Cloud round-trip p50 / p95 (ms) | ↓ |

**Auxiliary — 2D Rendering (diagnostic only, not a project goal):**

| Domain | Metric | Direction |
|--------|--------|-----------|
| 2D Rendering | PSNR | ↑ |
| 2D Rendering | SSIM | ↑ |
| 2D Rendering | LPIPS (AlexNet) | ↓ |

Report all metrics to 3 significant figures.
PSNR/SSIM/LPIPS reflect rendering quality, not planning utility. They are tracked for diagnostic purposes only.

### Quantitative Testing Convention (mandatory)

**Never rely on visual comparison of images alone.** Every test MUST output a JSON metrics file alongside any visualization. The E2E test (`eof3r/scripts/eval/test_e2e_pipeline.py`) demonstrates this pattern.

Five metric categories required in every pipeline test:

| Category | Key Metrics | Description |
|----------|-------------|-------------|
| **BEV Occupancy** | `bev_occupancy_coverage_t{0.1,0.3,0.5,0.7}`, `bev_occupancy_density`, `bev_spatial_extent_m2` | Coverage at multiple alpha thresholds |
| **Gaussian Quality** | `num_gaussians_total`, `opacity_mean/std/min/max`, `alpha_threshold_pass_rate`, `gaussian_spatial_range_{x,y,z}` | Per-Gaussian statistics |
| **Fusion Consistency** | `fg_bev_coverage`, `bg_bev_coverage`, `fg_bg_overlap_iou`, `drivable_occupancy_conflict_rate` | FG/BG overlap and conflict |
| **Costmap Validity** | `costmap_min/max`, `lethal_cell_count`, `free_cell_count`, `costmap_completeness` | Nav2 format correctness |
| **Timing** | `stage_times.{segmentation,background,foreground,fusion,costmap}`, `total_wall_time_s`, `peak_gpu_memory_mb` | Per-stage profiling |

Output convention:
- `outputs/eval/e2e_metrics.json` — all metrics as one flat dict
- `outputs/eval/e2e_pipeline_visualization.png` — supplementary visualization only

---

## §8 Thesis Materials

The `thesis/` directory stores curated, publication-quality materials — separate from `outputs/` which stores all raw experiment artifacts.

```
thesis/
├── writing/        # Paper drafts (.md → .tex). Gitignored.
├── figures/        # Publication-quality PDF/PNG, ≥300 DPI.
├── tables/         # LaTeX tables.
├── references/     # BibTeX + PDF papers.
└── notes/          # Writing-related thoughts.
```

### Figure Standards

- **DPI**: ≥300 for raster figures.
- **Format**: PDF for LaTeX, PNG for quick iteration.
- **Axes**: Clear labels, legible tick sizes.
- **Color**: Use unified palette from `eof3r/configs/plot_style.yaml` via `eof3r/src/utils/plotting.py`.
- **Legend**: Always visible, placed outside the plot area if needed.

### The key difference: `outputs/` is for everything; `thesis/` is for the best.

---

## §9 Git Workflow

### Branch Strategy

- `main` — always stable, runnable. Stage 0-3 work was committed directly to main (solo dev, fast iteration). For Stage 4+ (involving ROS2/Husky), use `stage/N` branches and squash-merge.

### Commit Messages

**English.** Format: `[stage/N] short description`

```
[stage/2] add ScanNet++ preprocessing script
[stage/3] fix NaN loss when training 3DGS on single object
[stage/5] implement alpha-blending fusion
```

### What NOT to Commit

- `data/` (except `data/test_fixtures/`)
- `outputs/`
- `baselines/` (except `registry.yaml` and `patches/`)
- `*.log`, `.env`, `__pycache__/`, `*.egg-info/`
- `thesis/writing/`
- Files >10MB (use download scripts instead)

### What to Commit

- `eof3r/src/`, `eof3r/scripts/`, `eof3r/configs/`, `docs/`, `CLAUDE.md`, `README.md`
- `eof3r/tests/`, `eof3r/.pre-commit-config.yaml`, `eof3r/pyproject.toml`, `eof3r/requirements.txt`

### Rules

- **Never rebase pushed commits.**
- **Never force-push to main.**
- **Always `git pull --rebase` before pushing.**

---

## §10 Environment & Dependencies

### Primary Environment

- **Name**: `eof3r`
- **Python**: 3.10
- **CUDA**: 12.1 (torch 2.5.1+cu121)
- **GPU**: NVIDIA RTX A6000 (48GB), peak VRAM ~6.3GB with VGGT + MVSplat loaded
- **Manager**: conda/mamba

### Activate

```bash
conda activate eof3r  # Python 3.10 + torch 2.5.1 + CUDA 12.1 + SAM2 + VGGT + MVSplat
```

### Dependency Layering

1. `eof3r/requirements.txt` — complete pip-installable dependencies
2. `environment.yml` — complete conda environment including non-Python deps (COLMAP, CUDA toolkits) [not yet created]
3. `baselines/registry.yaml` — per-baseline environments if isolation is needed

### GPU Target

- **Target VRAM**: <12GB (RTX 3060/4060 class).
- Every experiment must log GPU model and CUDA version.
- If a method requires more VRAM, document it and provide a downsampled alternative.

### Vehicle-Side Environment

- The Husky onboard computer (Jetson or x86) runs ROS2 Humble with its own environment.
- Vehicle-side code uses `rclpy` and standard ROS2 Python packages, installed via apt.
- The GPU training environment (`eof3r`) and vehicle environment are separate — do not mix them.
- Vehicle-side scripts live in `eof3r/scripts/robot/` and communicate with cloud via HTTP/gRPC (not ROS2 cross-network).

---

## §11 Reproducibility

- **Seed**: default 42, recorded in every experiment config.
- **Determinism**: use `torch.use_deterministic_algorithms(True)` where possible. If not possible, document why.
- **Config completeness**: experiment configs must contain all hyperparameters — no reliance on code defaults.
- **Dataset versioning**: record version/download date in `$EOF3R_DATA/registry.yaml`. Include checksums if feasible.
- **Randomness**: when uncontrollable randomness exists, run 3 times and report mean ± std.

---

## §12 Performance Principles

1. **Use pre-trained models whenever possible** — never re-train a baseline from scratch unless you have to.
2. **Cache intermediate results** in `outputs/cache/` — preprocessed features, extracted masks, etc.
3. **PyTorch DataLoader** with `num_workers>0` and `pin_memory=True` — don't let data loading be the bottleneck.
4. **Minimize CPU ↔ GPU transfers** — batch operations, avoid per-element loops.
5. **Use `torch.no_grad()` + `torch.cuda.amp`** in inference/eval mode.
6. **Profile before optimizing** — correctness first, then use `torch.profiler` to find real bottlenecks.
7. **Prefer vectorized operations** over Python loops. A single `einops` call beats a 10-line loop.
8. **Clean up GPU memory** — `del` large tensors when done, call `torch.cuda.empty_cache()` between stages.

---

## §13 Memory / Project Knowledge

> **"更新记忆" means updating this section.** Do NOT create separate memory files.
> This section records non-obvious project knowledge that isn't derivable from code or git.

### Network Proxy
When GitHub/pip connections fail (TLS error, timeout, clone failure, submodule fetch):
```bash
export http_proxy=http://192.168.213.103:53941
export https_proxy=http://192.168.213.103:53941
# For git: git config --global http.proxy http://192.168.213.103:53941
# For pip: pip install --proxy http://192.168.213.103:53941 <package>
```

### Environment Setup Notes
- **eof3r conda env**: Python 3.10, torch 2.5.1+cu121, RTX A6000 (48GB). All three real models verified.
- **MVSplat must be used from its own directory** (src/ path conflict with eof3r project src/). The wrapper handles this automatically via path isolation.
- **diff-gaussian-rasterization** must be installed with `--no-build-isolation` (needs torch at build time).
- **VGGT requires numpy<2** — pin `numpy==1.26.4` if conflicts arise.
- **MVSplat requires moviepy==1.0.3** (2.x API is incompatible).

### Key Design Decisions
- Subprocess approach was rejected for MVSplat; path isolation inside build() is cleaner.
- Vectorized fusion (scatter+gaussian_smooth) chosen over per-point loop (650x speedup).
- Stubs kept alongside real wrappers for CI/testing without GPU.

---

## Quick Reference: Key Documents

| Document | Purpose |
|----------|---------|
| `docs/project_scope.md` | What this project does (and does NOT do) |
| `docs/project_audit.md` | Diagnostic: current state vs new direction |
| `docs/roadmap.md` | 8-stage technical roadmap |
| `docs/lit_review.md` | Literature survey with reading checklist (12 directions) |
| `docs/experiments.md` | Navigation experiment designs (3 scenarios) |
| `docs/engineering.md` | Three-phase engineering plan |
| `docs/risks.md` | Risk register with mitigations and fallback paths |
| `docs/standards.md` | Supplementary detail for some standards |
| `docs/todo.md` | Task checklist, updated weekly |
| `docs/current_issues.md` | Active issues with root causes and solutions |
| `eof3r/configs/default.yaml` | Default configuration (full pipeline + robot + cloud) |
| `baselines/registry.yaml` | External baseline manifest |
