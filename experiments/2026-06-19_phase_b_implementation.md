# 实验日志：Phase B 训练基础设施实现

> 日期：2026-06-19
> 阶段：Phase B — Cross-Model Geometric Distillation

---

## 目标

实现 Phase B 的核心训练框架，为 ReSplat + VGGT-Ω 跨模型几何蒸馏做好准备。

## 完成工作

### 1. 模型获取
- **ReSplat** (github.com/cvg/ReSplat): MIT 许可, commit cc4594a
  - 16× fewer Gaussians (~8K vs 131K)
  - 需要 Python 3.12 + PyTorch 2.7.0 + CUDA 12.8（独立环境）
  - 使用 gsplat 渲染器
- **VGGT-Ω** (github.com/facebookresearch/vggt-omega): FAIR Noncommercial, commit 39a0cb8
  - Checkpoint 需要 HuggingFace 审批
  - +26% depth accuracy vs VGGT

### 2. 训练模块 (`eof3r/src/training/`)

#### 损失函数 (`losses.py`)
| 损失 | 公式 | 用途 |
|------|------|------|
| L_depth | 双向 Chamfer Distance | 高斯位置 ↔ VGGT 表面点 |
| L_occ | Focal Loss (γ=2) | 占据分类（处理 3.6% vs 96.4% 不平衡） |
| L_free | Squared Hinge (ε=0.05) | 自由空间正则化 |
| L_sem | Cross-Entropy | 语义分类 |
| L_color | L1 + SSIM (η=0.1) | 辅助光度损失 |

三阶段训练调度：
```
Stage 1 (0-30%):  α=1.0, β=0.3, γ=0.1, δ=0,   η=0.3  → 移动高斯到表面
Stage 2 (30-80%): α=0.5, β=1.0, γ=0.5, δ=0.3, η=0.1  → 占据+自由空间+语义
Stage 3 (80-100%): α=0.3, β=1.0, γ=1.0, δ=0.5, η=0.05 → 精细化
```

#### 占据/语义头 (`heads.py`)
- `OccupancyHead`: MLP (10→64→32→1), sigmoid 输出, 初始化 bias=-2.0
- `SemanticHead`: MLP (10→64→32→K), Xavier 初始化
- `ConfidenceHead`: MLP (10→32→16→1), 可选不确定性预测

#### VGGT 监督标注 (`supervision.py`)
- 逐高斯投影标记：Δd_i = μ̃_i^z - D^vggt(u_i, v_i)
- 自适应阈值：σ_i = κ · max_eig(Σ_i)
- 标记规则：|Δd|≤σ → OCCUPIED, Δd<-σ → FREE, Δd>σ → UNKNOWN
- 多视角标签合并

#### 训练器 (`trainer.py`)
- 差异化学习率：encoder (1e-4) vs heads (1e-3)
- OneCycleLR + cosine annealing
- 梯度裁剪 (0.5)
- 检查点保存（best + final）

### 3. 脚本
- `precompute_vggt_supervision.py`: 离线预计算 VGGT 监督数据
- `train_phase_b.py`: 主训练入口，支持 CLI override 和 mock dataloader

### 4. ReSplat Wrapper (`resplat_wrapper.py`)
- 遵循 MVSplatWrapper API 模式
- 路径隔离处理 ReSplat 独立环境
- 支持 HuggingFace checkpoint 加载

### 5. 测试
- 29/29 测试全部通过
- 覆盖：损失函数、头模块、监督标注、调度器

## 验证结果

```
All training module imports successful!
Chamfer loss: 0.5307
Focal loss: 0.2907
Hinge loss: 0.1981
Semantic CE loss: 2.7927
Step 0: stage=1, weights={'alpha': 1.0, 'beta': 0.3, 'gamma': 0.1, 'delta': 0.0, 'eta': 0.3}
Step 30000: stage=2, weights={'alpha': 0.5, 'beta': 1.0, 'gamma': 0.5, 'delta': 0.3, 'eta': 0.1}
Step 80000: stage=3, weights={'alpha': 0.3, 'beta': 1.0, 'gamma': 1.0, 'delta': 0.5, 'eta': 0.05}
Occupancy head output: torch.Size([50]), range [0.119, 0.119]
Semantic head output: torch.Size([50, 10])
All tests passed!

29 passed in 1.48s
```

## 下一步

1. **申请 VGGT-Ω checkpoint 访问权限**
   - URL: https://huggingface.co/facebook/VGGT-Omega
   - 需要 HuggingFace 账号审批

2. **创建 ReSplat 独立 conda 环境**
   ```bash
   conda create -y -n resplat python=3.12
   conda activate resplat
   pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r baselines/resplat/requirements.txt
   pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3
   cd baselines/resplat/src/model/encoder/pointops && python setup.py install
   ```

3. **运行 VGGT 监督预计算**
   ```bash
   conda activate eof3r
   python eof3r/scripts/preprocess/precompute_vggt_supervision.py --split train --max-scenes 100
   ```

4. **运行训练**
   ```bash
   # 先用 mock dataloader 验证流程
   python eof3r/scripts/train/train_phase_b.py --debug
   
   # 然后用真实数据
   python eof3r/scripts/train/train_phase_b.py --total-steps 100000
   ```

## 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| VGGT-Ω checkpoint 审批慢 | 无法使用最优 teacher | 先用现有 VGGT 做 teacher |
| ReSplat 环境不兼容 | 无法端到端训练 | 预计算+分离训练 |
| GPU 内存不足 | batch_size 受限 | 使用 ReSplat-small (76M) |

## 代码变更

```
baselines/registry.yaml          |  28 ++-
docs/todo.md                     |  19 +-
eof3r/configs/default.yaml       |  55 +++
eof3r/scripts/preprocess/        | 412 +++++++++++++++
eof3r/scripts/train/             | 310 +++++++++++
eof3r/src/foreground/            | 331 ++++++++++++
eof3r/src/training/              | 1653 ++++++++++++++++++++++++++++++
eof3r/tests/                     | 360 +++++++++++++
12 files changed, 3162 insertions(+)
```
