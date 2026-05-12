# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## §1 项目身份与语言

**EOF3R** — Efficient Object-level Feedforward 3D Reconstruction with 3D Gaussian Splatting.

本科毕业设计原型系统。将场景分解为前景物体（3DGS 精细重建）+ 背景区域（feedforward 模型快速重建）→ 融合为统一 3D 场景 → autonomous perception demo。

### 语言铁律

**只有两种东西用中文：**
1. 与用户对话
2. 项目文档文件（`docs/`、`lit_notes/`、`experiments/`、`README.md` 等面向人阅读的 markdown 文件）

**其余一切用英文：**
- Code comments, commit messages, variable names, function names, CLI arguments, log output, error messages, config field names, filenames

**模型名、方法名、学术术语保留原英文，不翻译。**

This file itself is in Chinese for §1 and English for code-facing sections — reflecting the two-zone rule.

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
│   ├── roadmap.md
│   ├── lit_review.md
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
│   └── model/                      # Model-specific hyperparameter configs
│
├── baselines/                      # External open-source code (gitignored)
│   ├── registry.yaml               # Manifest: URL, commit, env per baseline
│   └── patches/                    # Patches applied on top of upstream baselines
│
├── scripts/                        # Standalone scripts (runnable, not importable)
│   ├── download_data.sh
│   ├── preprocess/
│   └── eval/
│
├── src/                            # Core source code (importable Python package)
│   ├── __init__.py
│   ├── segmentation/               # Scene decomposition (SAM2 / YOLO wrapper)
│   ├── foreground/                 # Object-level 3DGS reconstruction
│   ├── background/                 # Feedforward background reconstruction (VGGT/DUSt3R wrapper)
│   ├── fusion/                     # Foreground-background fusion (core research)
│   ├── demo/                       # Autonomous perception demo
│   └── utils/                      # Shared utilities (IO, visualization, metrics)
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

- **Python files**: `snake_case.py` (e.g., `gaussian_renderer.py`)
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

- **Primary**: ScanNet++ (real indoor, camera poses included)
- **Backup**: Replica (synthetic, clean)
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

- **Right-handed, Y-up**, unit: **meters**.
- This is the OpenGL convention. If a baseline uses a different convention, document and convert at the wrapper boundary.

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

### Evaluation Metrics

| Domain | Metric | Direction |
|--------|--------|-----------|
| 2D Rendering | PSNR | ↑ |
| 2D Rendering | SSIM | ↑ |
| 2D Rendering | LPIPS (AlexNet) | ↓ |
| 3D Geometry | Chamfer Distance (L1) | ↓ |
| 3D Geometry | F-Score @1% / @5cm | ↑ |
| 3D Geometry | Normal Consistency | ↑ |
| Speed | Per-frame inference time (ms) | ↓ |

Report all metrics to 3 significant figures.

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

- `main` — always stable, runnable.
- `stage/N` — development branch for each stage. Squash-merge into `main` when the stage is done.
- Never commit directly to `main` (except Stage 0 initial setup).

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

- **Name**: `eof3r`
- **Python**: 3.10+
- **CUDA**: 12.x (fallback: 11.8)
- **Manager**: conda/mamba

### Dependency Layering

1. `requirements.txt` — minimal pip-installable dependencies (numpy, torch, open3d, etc.)
2. `environment.yml` — complete conda environment including non-Python deps (COLMAP, CUDA toolkits)
3. `baselines/registry.yaml` — per-baseline environments if isolation is needed

### GPU Target

- **Target VRAM**: <12GB (RTX 3060/4060 class).
- Every experiment must log GPU model and CUDA version.
- If a method requires more VRAM, document it and provide a downsampled alternative.

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
| `docs/roadmap.md` | 7-stage technical roadmap |
| `docs/lit_review.md` | Literature survey with reading checklist |
| `docs/standards.md` | Supplementary detail for some standards |
| `docs/todo.md` | Task checklist, updated weekly |
| `configs/default.yaml` | Default configuration |
| `baselines/registry.yaml` | External baseline manifest |
