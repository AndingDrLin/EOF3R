#!/usr/bin/env bash
set -euo pipefail

# Script is at eof3r/scripts/setup_mvsplat.sh — go up 2 levels to reach repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MVSPLAT_DIR="${MVSPLAT_DIR:-${REPO_ROOT}/baselines/mvsplat}"
ENV_PREFIX="${MVSPLAT_ENV_PREFIX:-${HOME}/lyj/anaconda3/envs/mvsplat}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
RASTER_URL="https://github.com/dcharatan/diff-gaussian-rasterization-modified"
RASTER_COMMIT="1250c420ebb945f0dce9945086e22faab9157c92"
RASTER_DIR="${MVSPLAT_RASTER_DIR:-${HOME}/.cache/eof3r/diff-gaussian-rasterization-modified}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set MVSPLAT_ENV_PREFIX or PYTHON_BIN to your mvsplat environment." >&2
  exit 1
fi

if [[ ! -f "${MVSPLAT_DIR}/requirements.txt" ]]; then
  echo "MVSplat requirements not found: ${MVSPLAT_DIR}/requirements.txt" >&2
  echo "Set MVSPLAT_DIR to the cloned mvsplat baseline path." >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install --upgrade pip wheel
"${PYTHON_BIN}" -m pip install 'setuptools<70'
"${PYTHON_BIN}" -m pip install \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
"${PYTHON_BIN}" -m pip install 'numpy<2'

TMP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${TMP_REQUIREMENTS}"' EXIT
grep -v 'diff-gaussian-rasterization-modified' "${MVSPLAT_DIR}/requirements.txt" > "${TMP_REQUIREMENTS}"
"${PYTHON_BIN}" -m pip install -r "${TMP_REQUIREMENTS}"
"${PYTHON_BIN}" -m pip install 'numpy<2'

mkdir -p "$(dirname "${RASTER_DIR}")"
if [[ ! -d "${RASTER_DIR}/.git" ]]; then
  rm -rf "${RASTER_DIR}"
  git -c http.version=HTTP/1.1 clone "${RASTER_URL}" "${RASTER_DIR}"
fi

git -C "${RASTER_DIR}" fetch origin "${RASTER_COMMIT}" || true
git -C "${RASTER_DIR}" checkout "${RASTER_COMMIT}"
git -C "${RASTER_DIR}" submodule update --init --recursive
"${PYTHON_BIN}" -m pip install --no-build-isolation "${RASTER_DIR}"

"${PYTHON_BIN}" - <<'PY'
import numpy
import torch
from diff_gaussian_rasterization import GaussianRasterizer

print(f"numpy {numpy.__version__}")
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, cuda_available {torch.cuda.is_available()}")
print(f"diff_gaussian_rasterization {GaussianRasterizer.__name__}")
PY
