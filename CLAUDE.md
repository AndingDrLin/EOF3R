# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## §1 项目身份与语言

**EOF3R** — Efficient Object-level Feedforward 3D Reconstruction with 3DGS.

**Despite the name, the project produces planning-oriented Gaussian occupancy, not photorealistic images.**

本科毕业设计原型系统。面向 Husky 低速无人车在校园/园区的"最后 50 米"配送场景。融合 G2O 几何约束思想的前馈式 object-level Gaussian occupancy 方法：

物体分离（SAM2）→ 背景 3R 粗几何估计（VGGT/MASt3R）→ 前景 G2O-inspired feedforward Gaussian occupancy 预测（occupancy_alpha, footprint, semantic, confidence）→ 融合 → BEV semantic costmap → 本地规划器避障与路径选择。

**系统不追求逼真重建，而追求更可靠、更适合规划的几何-语义表示。**

**系统架构**：车端始终运行本地安全回路（相机/里程计/IMU/急停/Nav2 局部规划/cmd_vel），云端负责高算力异步推理（SAM2 分割细化 / 3R 背景几何估计 / G2O-inspired feedforward Gaussian occupancy / 语义 costmap 生成）。云端结果是 planning enhancement，不直接控制车辆。云端返回的是 lightweight planning-oriented representation（object state, 3D bbox, BEV footprint, semantic label, risk score, confidence, costmap patch），不是完整 Gaussian 渲染模型。

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
# End-to-end pipeline test (with full quantitative metrics)
conda activate mvsplat  # workaround: eof3r env not yet created
python scripts/eval/test_e2e_pipeline.py --skip-mvsplat
# Omit --skip-mvsplat on GPU machines with MVSplat checkpoint available.
# Outputs: outputs/eval/e2e_metrics.json + e2e_pipeline_visualization.png

# BEV projection verification (MVSplat only)
conda activate mvsplat
cd baselines/mvsplat && python ../../scripts/eval/verify_mvsplat_bev.py

# Lint & format (auto-fix)
ruff check --fix src/ scripts/
ruff format src/ scripts/

# Pre-commit (runs ruff + trailing-whitespace + large-file check)
pre-commit run --all-files

# Setup a baseline (example: mvsplat)
bash scripts/setup_mvsplat.sh
# Each baseline has its own setup script and conda env (see baselines/registry.yaml)
```

### Environment

```bash
# Single unified environment (2025-06-18)
conda activate eof3r  # Python 3.10 + torch 2.5.1 + CUDA 12.1 + all models

# To recreate from scratch:
conda create -n eof3r python=3.10 -y && conda activate eof3r
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e baselines/sam2/
pip install -e baselines/vggt/
pip install -e baselines/mvsplat/  # not a pip pkg — install deps manually
# For full deps list, inspect the install commands in memory or docs/current_issues.md.

# GPU: NVIDIA RTX A6000 (48GB), peak VRAM ~6.3GB (VGGT + MVSplat together)
```

### Known Blockers (as of 2025-06-18)

- ~~SAM2/VGGT cannot be cloned~~ → **Resolved.** Cloned via network proxy (192.168.213.103:53941).
- ~~eof3r env not created~~ → **Resolved.** All three real models (SAM2, VGGT, MVSplat) verified in eof3r.
- **Only remaining blocker**: no campus rosbag. Public Re10k dataset used as substitute for E2E validation.

### Config-driven Experiments

All experiments are driven by YAML configs inheriting from `configs/default.yaml`. Override fields via CLI or experiment-specific YAML. Never hardcode hyperparameters in code.

---

## §1c Pipeline Architecture

The system runs a **4-stage pipeline**. Each stage maps to a `src/` module:

```
Input: RGB images + camera intrinsics
       │
       ▼
┌─────────────────┐
│  segmentation/   │  SAM2 / YOLO → per-object masks + class labels
└────────┬────────┘
         │ foreground masks, background region
         ├──────────────────────────────┐
         ▼                              ▼
┌─────────────────┐           ┌─────────────────┐
│  foreground/     │           │  background/     │
│  MVSplat / 3DGS  │           │  VGGT / MASt3R  │
│  → per-object    │           │  → coarse pointmap
│  Gaussian        │           │    + ground plane
│  occupancy       │           │    + traversable
└────────┬────────┘           └────────┬────────┘
         │                              │
         └──────────┬───────────────────┘
                    ▼
           ┌─────────────────┐
           │  fusion/         │  Align coords (Y-up → Z-up), BEV projection,
           │                  │  Gaussian → occupancy grid
           └────────┬────────┘
                    ▼
           ┌─────────────────┐
           │  costmap/        │  BEV semantic costmap → ROS2 Nav2
           │                  │  (inflation, semantic weights, risk scores)
           └────────┬────────┘
                    ▼
           ┌─────────────────┐
           │  communication/  │  Vehicle ↔ cloud async bridge
           └─────────────────┘
```

**Key interfaces between modules (actual class names):**
- `segmentation` → `foreground`: object masks (SAM2Stub → MVSplatWrapper)
- `segmentation` → `background`: background region mask (SAM2Stub → VGGTStub)
- `foreground` / `background` → `fusion`: aligned 3D Gaussians (numpy) / pointmaps in shared Y-up coords
- `fusion` → `costmap`: BEV occupancy grid in Z-up robot frame (BEVProjector → CostmapGenerator)
- `costmap` → ROS2: uint8 costmap array (0=free, 254=lethal), not yet published to ROS topic

**Stub vs Real Status (as of 2025-06-18):**
| Module | File | Status |
|--------|------|--------|
| segmentation | `sam2_stub.py` | 🟡 Stub (SAM2 blocked by GitHub TLS) |
| foreground | `mvsplat_wrapper.py` | 🟢 Real MVSplat wrapper (build/infer/extract_occupancy) |
| background | `vggt_stub.py` | 🟡 Stub (VGGT blocked by GitHub TLS) |
| fusion | `bev_projector.py`, `coord_utils.py` | 🟢 Real (Y-up→Z-up, BEV projection, FG/BG fusion) |
| costmap | `costmap_generator.py` | 🟢 Real (Nav2 uint8 format, semantic weights, inflation) |
| communication | `__init__.py` only | 🔴 Empty stub |

---

## §2 Directory Structure & Naming

```
EOF3R/
├── README.md                       # Project overview & current progress
├── CLAUDE.md                       # This file — project constitution
├── requirements.txt                # Minimal Python dependencies
├── pyproject.toml                  # Project metadata & ruff config
├── .pre-commit-config.yaml         # Pre-commit hooks
├── .gitignore
├── .env.example                    # Environment variable template
│
├── docs/                           # Project documentation (中文)
│   ├── project_scope.md
│   ├── project_audit.md             # Diagnostic: current state vs new direction
│   ├── roadmap.md
│   ├── lit_review.md
│   ├── experiments.md               # Navigation experiment designs
│   ├── engineering.md               # Three-phase engineering plan
│   ├── risks.md                     # Risk register with mitigations
│   ├── standards.md
│   └── todo.md
│
├── lit_notes/                      # Paper reading notes (中文, one per paper)
│   └── _template.md
│
├── experiments/                    # Experiment logbook (中文, one per experiment)
│   └── exp_template.md
│
├── configs/                        # YAML configuration files
│   ├── default.yaml                # Default config — all experiments inherit from this
│   ├── plot_style.yaml             # Unified plotting style for paper figures
│   ├── data/                       # Dataset-specific configs
│   ├── model/                      # Model-specific hyperparameter configs
│   ├── robot/                      # Husky/Nav2 parameter configs
│   └── cloud/                      # Cloud server configs
│
├── baselines/                      # External open-source code (gitignored)
│   ├── registry.yaml               # Manifest: URL, commit, env per baseline
│   └── patches/                    # Patches applied on top of upstream baselines
│
├── scripts/                        # Standalone scripts (runnable, not importable)
│   ├── preprocess/                 # [empty]
│   ├── eval/test_e2e_pipeline.py   # Full 5-stage E2E test with quantitative metrics
│   ├── eval/verify_mvsplat_bev.py  # MVSplat → BEV verification
│   └── robot/                      # Husky launch/ROS2 config scripts [.gitkeep only]
│
├── src/                            # Core source code (importable Python package)
│   ├── __init__.py
│   ├── segmentation/               # Scene decomposition (SAM2Stub → future SAM2Wrapper)
│   ├── foreground/                 # MVSplatWrapper — feedforward Gaussian occupancy
│   ├── background/                 # VGGTStub → future VGGTWrapper
│   ├── fusion/                     # BEVProjector + coord_utils (Y-up↔Z-up)
│   ├── costmap/                    # CostmapGenerator (Nav2 uint8 format)
│   ├── communication/              # [stub only] Vehicle-cloud async communication
│   ├── demo/                       # [empty] ROS2 navigation demo
│   └── utils/                      # [empty] Shared utilities
│
├── data/                           # NOT version-controlled
│   ├── test_fixtures/              # Small test data (<1MB, committed)
│   ├── raw/                        # Symlinks to $EOF3R_DATA (gitignored)
│   └── processed/                  # (gitignored)
│
├── outputs/                        # NOT version-controlled
│   ├── results/{exp_name}/         # Experiment outputs: config.yaml, metrics.json,
│   │                               #   visualizations/, checkpoints/
│   ├── logs/{exp_name}.log         # Training logs
│   └── cache/                      # Cached intermediate results
│
├── thesis/                         # Paper-writing materials (separate from outputs)
│   ├── writing/                    # Drafts (.md / .tex) — gitignored
│   ├── figures/                    # Publication-quality PDF/PNG (≥300 DPI)
│   ├── tables/                     # LaTeX tables
│   ├── references/                 # BibTeX / PDF references
│   └── notes/                      # Writing-related notes
│
└── tests/                          # Smoke tests per module
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

3. **Wrap every baseline behind a Python interface** in the corresponding `src/` module. Example:
   - `src/foreground/gaussian_splatting_wrapper.py` wraps `baselines/gaussian-splatting/`
   - `src/background/vggt_wrapper.py` wraps `baselines/vggt/`
   - All wrappers expose a consistent API (`build()`, `train()`, `infer()`, `save()`, `load()`)

4. **Isolate environments** — each baseline may have its own conda env (recorded in registry.yaml). Provide a setup script in `scripts/setup_{baseline}.sh`.

5. **Pre-trained weights** go to `outputs/checkpoints/{baseline_name}/`, not inside `baselines/`.

6. **Baseline code is gitignored.** Only `registry.yaml` and `patches/` are committed.

### Current Status (2025-06-18)

| Baseline | Status | Notes |
|----------|--------|-------|
| MVSplat | 🟢 Cloned + checkpoints | re10k.ckpt, acid.ckpt |
| DepthSplat | 🟢 Cloned | Not yet used, not in registry.yaml |
| SAM2 | 🔴 Blocked | GitHub TLS handshake failure — using SAM2Stub |
| VGGT | 🔴 Blocked | GitHub TLS handshake failure — using VGGTStub |
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

- Switch datasets via `configs/default.yaml` → `data.dataset`, never hardcode dataset names in code.
- Preprocessing scripts live in `scripts/preprocess/{dataset_name}.py`.
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

- `scripts/` — standalone runnable scripts. Must have `if __name__ == "__main__":` guard.
- `src/` — importable modules. No side effects on import.

### Testing

- Smoke tests in `tests/` — verify each module's pipeline can run end-to-end on a small fixture.
- Naming: `tests/test_{module}.py`
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

**Never rely on visual comparison of images alone.** Every test MUST output a JSON metrics file alongside any visualization. The E2E test (`scripts/eval/test_e2e_pipeline.py`) demonstrates this pattern.

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
- **Color**: Use unified palette from `configs/plot_style.yaml` via `src/utils/plotting.py`.
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

- `src/`, `scripts/`, `configs/`, `docs/`, `lit_notes/`, `experiments/`
- `tests/`, `.pre-commit-config.yaml`, `pyproject.toml`

### Rules

- **Never rebase pushed commits.**
- **Never force-push to main.**
- **Always `git pull --rebase` before pushing.**

---

## §10 Environment & Dependencies

### Primary Environment

- **Name**: `eof3r` (not yet created — workaround: use `mvsplat` or `depthsplat` conda envs)
- **Python**: 3.10+
- **CUDA**: 11.8 (mvsplat env) / 12.x (target for eof3r)
- **Manager**: conda/mamba

### Current Workaround

Until `eof3r` env is created, use baseline conda envs:
```bash
conda activate mvsplat     # torch 2.1.2+cu118 — used for all current dev
conda activate depthsplat  # alternative
pip install pyyaml scipy matplotlib  # add missing deps as needed
```

### Dependency Layering

1. `requirements.txt` — minimal pip-installable dependencies (numpy, torch, open3d, etc.)
2. `environment.yml` — complete conda environment including non-Python deps (COLMAP, CUDA toolkits)
3. `baselines/registry.yaml` — per-baseline environments if isolation is needed

### GPU Target

- **Target VRAM**: <12GB (RTX 3060/4060 class).
- Every experiment must log GPU model and CUDA version.
- If a method requires more VRAM, document it and provide a downsampled alternative.

### Vehicle-Side Environment

- The Husky onboard computer (Jetson or x86) runs ROS2 Humble with its own environment.
- Vehicle-side code uses `rclpy` and standard ROS2 Python packages, installed via apt.
- The GPU training environment (`eof3r`) and vehicle environment are separate — do not mix them.
- Vehicle-side scripts live in `scripts/robot/` and communicate with cloud via HTTP/gRPC (not ROS2 cross-network).

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
| `configs/default.yaml` | Default configuration (full pipeline + robot + cloud) |
| `baselines/registry.yaml` | External baseline manifest |
