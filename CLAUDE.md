# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## §1 项目身份与语言

**EOF3R** — Efficient Object-level Feedforward 3D Reconstruction with 3DGS.

**核心创新**：将 feedforward 3DGS 从 photorealistic 渲染工具改造为 planning-oriented 几何-语义占据预测器。方法是**跨模型几何蒸馏（cross-model geometric distillation）**——用前馈几何模型（VGGT-Ω）的输出作为前馈 3DGS 模型（ReSplat）的训练监督信号，通过概率占据场建模→负对数似然→可计算损失函数的严谨推导，使 3DGS 学会预测 metric-scale 占据而非逼真颜色。

**Despite the name, the project produces planning-oriented Gaussian occupancy, not photorealistic images.**

本科毕业设计原型系统。面向 Husky 低速无人车在校园/园区的"最后 50 米"配送场景。

**方法本质**：
- **训练时**：VGGT-Ω 提供 depth + pointmap + free-space rays → 几何监督；YOLO+SAM2 提供 2D masks → 语义监督。ReSplat 学习用 Gaussian primitives 预测 occupancy + semantic + confidence + free-space（而非 opacity + SH + color）
- **推理时**：只需要 ReSplat（单模型，前馈）→ 直接从 RGB 输出 metric-scale BEV 占据 + 语义 costmap
- VGGT-Ω 是 frozen geometry teacher，不是 inference stage——与"拼接方案"的本质区别
- 2026-06-19 Occupancy Head POC 实验证明：post-hoc MLP 在现有 Gaussians 上无效（仅 2.6% 靠近真实表面），必须端到端重训 decoder + Gaussian positions

**系统架构**：车端始终运行本地安全回路（相机/里程计/IMU/急停/Nav2 局部规划/cmd_vel），云端运行改造后的 ReSplat（单模型前馈推理）。推理时只跑 ReSplat，输出 BEV occupancy + semantic costmap。总延迟目标 <10s（VGGT-Ω ~5s + ReSplat ~2s + BEV ~0.01s）。

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
# === Verification (Phase A — confirmed working) ===
# E2E pipeline test (all real models): 21.4s, coverage 1.88%, IoU 0.0047
source ~/anaconda3/etc/profile.d/conda.sh && conda activate eof3r
python eof3r/scripts/eval/test_e2e_pipeline.py

# Ablation study (4 variants × 3 frame pairs)
python eof3r/scripts/eval/ablation_study.py

# Occupancy Head POC (post-hoc MLP experiment — Phase A.1)
python eof3r/scripts/eval/test_occupancy_head.py

# BEV projection verification
python eof3r/scripts/eval/verify_mvsplat_bev.py

# === Code quality ===
ruff check --fix eof3r/src/ eof3r/scripts/ --config eof3r/pyproject.toml
ruff format eof3r/src/ eof3r/scripts/ --config eof3r/pyproject.toml

# === Setup ===
bash eof3r/scripts/setup_mvsplat.sh
```

### Environment

```bash
# Single unified environment (updated 2026-06-19)
conda activate eof3r  # Python 3.10 + torch 2.5.1 + CUDA 12.1 + all models
# For non-interactive shells (CI/scripts): source ~/anaconda3/etc/profile.d/conda.sh && conda activate eof3r

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

### Known Blockers (updated 2026-06-19)

- ~~环境搭建、模型安装、数据准备~~ → **全部 Resolved。**
- 🔴 **核心 Blocker**：MVSplat/ReSplat 的 Gaussian primitives 为 photorealistic 渲染优化→BEV 不可用。三个机制性失败（见 §1c）：
  1. **Opacity≠Occupancy**：α 是 alpha-blending 渲染权重，非占据概率。POC 实验证明仅 2.6% Gaussians 靠近 VGGT 表面，68.9% 在自由空间。
  2. **Covariance Loss**：scatter+smooth BEV 投影丢弃 Σ 结构→各向同性过膨胀。
  3. **No Free-Space**：VGGT pointmap 只给表面点→无法区分 free/occ/unknown。
  - 解决方向：概率占据场建模→VGGT-Ω 几何监督→端到端重训 ReSplat decoder（见 §1c）。
- **次要 Blocker**：无校园 rosbag。公开数据集 Re10k 作为验证替代。
- **Conda 注意事项**：非交互 shell 需 `source ~/anaconda3/etc/profile.d/conda.sh && conda activate eof3r`。交互终端开箱即用。

### Config-driven Experiments

All experiments are driven by YAML configs inheriting from `eof3r/configs/default.yaml`. Override fields via CLI or experiment-specific YAML. Never hardcode hyperparameters in code.

---

## §1c Architecture: Cross-Model Geometric Distillation

### Three Mechanistic Failures (Confirmed by Phase A + POC)

| # | Failure | Mechanism | Quantitative Evidence (2026-06-19) |
|---|---------|-----------|-----------------------------------|
| 1 | **Opacity ≠ Occupancy** | α is alpha-blending weight entangled with color. Low α+high SH = same pixel as high α+low SH. | α_mean=0.28, pass_rate(>0.5)=2.5%, POC: 2.6% Gaussians near surface, 68.9% in free space |
| 2 | **Covariance Loss** | BEV scatter+smooth discards Σ→anisotropic→isotropic inflation by `3·max(scale)` | Fixed-grid coverage=1.88%, dynamic-grid=85.5% (self-adaptive artifact) |
| 3 | **No Free-Space** | VGGT pointmap surfaces all→occupied. No FREE/OCCUPIED/UNKNOWN distinction | Costmap lethal=55%, free=42%, cannot distinguish free from unknown |

### Architecture: VGGT as Teacher, ReSplat as Student

> **Teacher 选型**：当前使用原版 VGGT（已验证，13.6s）。论文最终阶段切换到 VGGT-Ω（+26% depth 精度）。
> 切换不影响方法论，只影响监督信号质量。

```
                    TRAINING                              │        INFERENCE
                                                          │
  ┌──────┐    ┌─────────┐                                │   ┌──────┐
  │ SAM2 │    │  VGGT   │  ← frozen teachers             │   │ RGB  │
  └──┬───┘    └────┬────┘                                │   └──┬───┘
     │2D masks     │depth, pointmap, free-space rays     │      │
     │             │                                      │      ▼
     ▼             ▼                                      │   ┌──────────┐
  ┌────────────────────────────────┐                      │   │ ReSplat  │
  │          ReSplat               │  ← train: decoder    │   │ (infer)  │
  │  freeze: encoder               │    + Gaussian adapter│   └───┬──────┘
  │  train:  occupancy head        │    + occupancy head  │       │
  │          semantic head         │    + semantic head   │       ▼
  │          Gaussian positions    │                      │   ┌──────────┐
  └────────────────────────────────┘                      │   │   BEV    │
                                                          │   │occupancy │
  L_total = α·L_depth + β·L_occ + γ·L_free + δ·L_sem     │   │+semantic │
            + η·L_color  (η=0.1, auxiliary)               │   │+costmap  │
                                                          │   └──────────┘
```

### Loss Function (Probabilistic Derivation — see `docs/lit_notes/phaseb_design_2026-06-19.md` §2)

Each Gaussian defines an occupancy field: $p_i(\mathbf{x}) = o_i \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$

VGGT provides per-pixel depth $D^{\text{vggt}}$ and pointmap $\mathcal{P}^{\text{vggt}}$.  Per-Gaussian labeling via projection to VGGT camera:

$$\Delta d_i = \tilde{\mu}_i^z - D^{\text{vggt}}(\pi(\tilde{\boldsymbol{\mu}}_i)), \quad \sigma_i = \kappa \cdot \max\text{eig}(\boldsymbol{\Sigma}_i)$$

$$y_i = \begin{cases} 1 & |\Delta d_i| \leq \sigma_i \text{ (OCCUPIED)} \\ 0 & \Delta d_i < -\sigma_i \text{ (FREE)} \\ \text{mask} & \Delta d_i > \sigma_i \text{ (UNKNOWN)} \end{cases}$$

| Loss | Formula | Purpose |
|------|---------|---------|
| $\mathcal{L}_{\text{depth}}$ | $\frac{1}{\|\mathcal{P}\|}\sum_{\mathbf{p}}\min_i\|\boldsymbol{\mu}_i-\mathbf{p}\|^2 + \frac{1}{\|\mathcal{O}\|}\sum_{i\in\mathcal{O}}\min_{\mathbf{p}}\|\boldsymbol{\mu}_i-\mathbf{p}\|^2$ | Chamfer: Gaussian means ↔ VGGT surfaces |
| $\mathcal{L}_{\text{occ}}$ | $-\frac{1}{\|\mathcal{L}\|}\sum_i [w_1 y_i(1-o_i)^\gamma\log o_i + w_0(1-y_i)o_i^\gamma\log(1-o_i)]$ | Focal Loss ($\gamma=2$): occupy/free classification |
| $\mathcal{L}_{\text{free}}$ | $\frac{1}{\|\mathcal{F}\|}\sum_{i\in\mathcal{F}}\max(0, o_i-\epsilon)^2, \epsilon=0.05$ | Squared hinge: free-space Gaussians → low occupancy |
| $\mathcal{L}_{\text{sem}}$ | $-\frac{1}{\|\mathcal{O}\|}\sum_{i\in\mathcal{O}}\log\text{softmax}(\mathbf{s}_i)_{c_i}$ | Per-Gaussian semantic classification |
| $\mathcal{L}_{\text{color}}$ | $\eta\cdot\frac{1}{HW}\sum[|I_{\text{rend}}-I_{\text{gt}}|_1 + 0.2(1-\text{SSIM})]$ | Auxiliary only ($\eta=0.1$): prevent encoder drift |

### Three-Stage Training Schedule

```
Stage 1 (Warmup, ~30%):  α=1.0 β=0.3 γ=0.1 δ=0   η=0.3  → Gaussians move to surfaces
Stage 2 (Main, ~50%):    α=0.5 β=1.0 γ=0.5 δ=0.3 η=0.1  → occ + free-space + semantics
Stage 3 (Fine, ~20%):    α=0.3 β=1.0 γ=1.0 δ=0.5 η=0.05 → refine, color exits
```

### Implementation Roadmap

- **Phase A** ✅ (2026-06-19): Sequential baseline + 4-variant ablation. Confirmed 3 failure modes. IoU=0.052, cov=1.88%, lethal=55%.
- **Phase A.1** ✅ (2026-06-19): Occupancy Head POC. Proved post-hoc MLP insufficient — only 2.6% Gaussians near VGGT surfaces. Must retrain Gaussian positions end-to-end.
- **Phase B** 🔜: ReSplat decoder retraining with VGGT-Ω geometric supervision. Replace MVSplat wrapper with ReSplat; add occupancy/semantic heads; implement $\mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{occ}} + \mathcal{L}_{\text{free}}$; 3-stage training on Re10k.
- **Phase C**: Differentiable BEV marginalization — analytical Σ→XZ projection; ray-based free-space carving.
- **Phase D**: End-to-end planning loss; RL-based adaptive Gaussian density allocation.

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

### Current Status (2026-06-19)

| Baseline | Status | Notes |
|----------|--------|-------|
| MVSplat | 🟢 Phase A verified | re10k.ckpt, 131K Gaussians, α_mean=0.28. Phase B targets replacement with ReSplat |
| DepthSplat | 🟢 Cloned | MVSplat successor. DepthAnythingV2 fusion. Backup student model |
| ReSplat | ⬜ Research | Preferred Phase B student (16× fewer Gaussians, recurrent refinement). Not yet cloned |
| CoSplat | ⬜ Research | Backup student (tri-plane consensus, best geometric consistency) |
| SAM2 | 🟢 Verified | YOLOv8-nano frontend → 3 objects with real COCO labels |
| VGGT | 🟢 Phase A verified | 1B model, ~13.6s. Scale recovery ×7.8 via ground plane |
| VGGT-Ω | ⬜ Research | CVPR 2026 Oral. Depth δ1.25=93.5% (+26% vs VGGT), 1.6× faster. Phase B teacher |
| YOLOv8 | 🟢 Integrated | ultralytics pip, 6MB nano model |
| DUSt3R, MASt3R | ⬜ Not started | — |
| Nav2 | ⬜ Not started | Husky only |

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
- Subprocess approach rejected for MVSplat; path isolation inside build() is cleaner.
- Vectorized fusion (scatter+gaussian_smooth) over per-point loop (650x speedup).
- Stubs kept alongside real wrappers for CI/testing without GPU.
- YOLOv8-nano (6MB) → SAM2 box-prompt (65→3 objects, real COCO semantics, 2.5× speedup).
- Dynamic BEV grid (auto bounds) via `set_bounds_from_points()` — prevents shape mismatch.
- **Phase B design (2026-06-19)**: VGGT teacher (原版，最终切换 Ω) + ReSplat student (16× fewer Gaussians), probabilistic occupancy-field loss derivation, 3-stage training, RL for Gaussian density allocation, PBT for hyperparams.

### Conda in Non-Interactive Shells
- Non-interactive shells skip `.bashrc` → use `source ~/anaconda3/etc/profile.d/conda.sh && conda activate eof3r`

### Phase A Verified Results (2026-06-19)
- **E2E**: 21.4s total, 131K Gaussians, α_mean=0.28, fixed-grid coverage=1.88%
- **Ablation** (4×3): A_full IoU=0.052, scale=7.8; B_noscale IoU=0, scale=7018; C_noalign IoU=0; D_auto 69 objects
- **Conclusion**: Scale+alignment necessary but insufficient. Three failure modes bottleneck BEV.

### Phase A.1 POC (2026-06-19)
- Post-hoc MLP occupancy head: VGGT depth projection→per-Gaussian labels (2.6% occ, 68.9% free, 28.5% unknown)
- MLP val acc=96.5% but BEV coverage <1% for all methods (opacity/VGGT labels/MLP)
- **Proved**: Gaussian positions are the problem, not opacity prediction. Must retrain decoder end-to-end.

### Phase B Design (see `docs/lit_notes/phaseb_design_2026-06-19.md` for full derivation)
- **Teacher**: VGGT (原版，已验证 13.6s)。最终阶段切换 VGGT-Ω (CVPR 2026 Oral, +26% depth)。
- **Student**: ReSplat (16× fewer Gaussians, recurrent refinement) preferred; CoSplat (tri-plane consensus) backup.
- **Loss**: Probabilistic occupancy field→NLL→Chamfer+Focal+Hinge+CE+L1. 3-stage training schedule.
- **AutoLab confirmed**: Focal loss 3.5× > BCE, 3-stage schedule 20.6% > uniform, 30K steps sufficient.
- **Hyperparams**: Optuna (initial)→PBT (adaptive)→BO (fine). RL for per-region Gaussian density allocation.
- **Inference target**: <10s total (VGGT ~14s → ReSplat ~2s + BEV ~0.01s; Ω upgrade: ~5s + 2s + 0.01s)

### Phase B Implementation Status (2026-06-19)
- **Training module implemented**: `eof3r/src/training/` — losses, heads, supervision, trainer, train script
- **AutoLab 8 experiments complete** (mock data): pipeline verified, focal loss 3.5× > BCE, stage schedule 20.6% > uniform
- **ReSplat cloned**: `baselines/resplat/` (MIT, cc4594a). Needs Python 3.12 + PyTorch 2.7.0 + CUDA 12.8.
- **VGGT-Ω cloned**: `baselines/vggt-omega/` (FAIR Noncommercial, 39a0cb8). Checkpoint gated on HuggingFace.
- **Teacher decision**: Use original VGGT for now (verified, 13.6s). Switch to VGGT-Ω at final stage.
- **Conda path**: `/home/ubuntu/lyj/anaconda3/` (NOT `~/anaconda3/`). Non-interactive: `source /home/ubuntu/lyj/anaconda3/etc/profile.d/conda.sh`
- **29/29 tests passing** in `eof3r/tests/test_training.py`
- **Next steps**: (1) VGGT supervision pre-computation on Re10k, (2) Load real ReSplat encoder, (3) Train with real data, (4) Evaluate with real metrics
- **ReSplat env isolation**: ReSplat needs separate env. Pre-compute VGGT supervision in eof3r env, train in resplat env.

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
