# Phase B 深度设计：后端选型、损失函数数学推导、RL 超参优化

> 日期：2026-06-19
> 触发：用户要求 (a) 替换更强后端模型, (b) 损失函数严谨数学推导, (c) RL 超参学习

---

## 一、MVSplat 后端替换分析

### 2025-2026 Feedforward 3DGS 全景对比

| 方法 | 核心机制 | 几何精度 | 高斯数量 | 推理速度 | 适合占据改造？ |
|------|---------|---------|---------|---------|--------------|
| **DepthSplat** (CVPR 2025) | Cost volume + DepthAnythingV2 | 中（深度更准但仍是 photometric-first） | ~131K (per-pixel) | ~0.6s (A100) | ⭐⭐⭐ 深度更好→位置更准 |
| **ReSplat** (2025) | Recurrent refinement, 16× 压缩 | 中高（压缩后仍保持精度） | **~8K** (subsampled) | ~0.02s | ⭐⭐⭐⭐ 高斯极稀疏→BEV投影极快 |
| **CoSplat** (2026) | Tri-plane 共识 + confidence-aware | **高**（显式过滤几何 outlier） | per-pixel | 3× faster than graph | ⭐⭐⭐⭐ 几何一致性最强 |
| **ZipSplat** (Jun 2026) | Token-based, k-means 解耦像素 | 中高 | **~6× fewer** | 单模型多质量档 | ⭐⭐⭐⭐⭐ 高斯数与场景复杂度挂钩 |
| **AnchorSplat** (CVPR 2026) | 3D anchors + FPS 采样 | 高（anchor约束减少floater） | ~247K | 3.1-6.1s | ⭐⭐⭐ 锚点天然适合VGGT |
| **AirSplat** (2026) | 3D VFM prior + rating-based opacity | 中高 | per-pixel | — | ⭐⭐⭐ VFM prior可替换为VGGT |

### 推荐：ReSplat 或 CoSplat

**首选 ReSplat**（如果开源可用）：
- **16× 更少高斯**（~8K vs 131K）→ BEV 投影计算量从 O(N×r²) 降为 O(N/16×r²)
- Recurrent refinement 用 rendering error 做反馈信号——**可以直接替换为 VGGT geometric error**
- 作者是 MVSplat/DepthSplat 同一团队（Haofei Xu），代码质量有保障

**次选 CoSplat**（如果需要更强的几何一致性）：
- Tri-plane 全局共识显式过滤几何不一致的高斯
- Confidence-aware splatting 可以改造为 occupancy-aware splatting
- 但代码较新，可能需要更多适配工作

**不推荐**：纯 per-pixel 方法（MVSplat/DepthSplat/AA-Splat）——Gaussian 数量固定于像素数，无法按场景几何复杂度自适应分配。

---

## 二、损失函数的严谨数学推导

### 2.1 问题形式化

**符号定义**：
- 输入图像：$\mathcal{I} = \{I_1, \ldots, I_V\}$, $I_v \in \mathbb{R}^{3 \times H \times W}$
- VGGT-Ω 输出（frozen teacher）：
  - 深度图 $D^{\text{vggt}}(u,v) \in \mathbb{R}^+$
  - 点云 $\mathcal{P}^{\text{vggt}} = \{\mathbf{p}_j \in \mathbb{R}^3\}_{j=1}^{M}$
  - 相机位姿 $\mathbf{T}_v \in SE(3)$，内参 $\mathbf{K}_v$
- MVSplat 输出（student，需训练）：
  - 高斯参数 $\mathcal{G} = \{(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i, \mathbf{h}_i)\}_{i=1}^{N}$
  - 额外输出头：占据值 $o_i = f_\theta(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i) \in [0,1]$
  - 语义 logits $\mathbf{s}_i = g_\phi(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \mathbf{h}_i) \in \Delta^{K-1}$

**核心假设**：每个高斯球定义了一个三维占据概率场：
$$p_i(\mathbf{x}) = o_i \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$

场景中任意点 $\mathbf{x}$ 被占据的概率（union of Gaussians，取一阶近似）：
$$p_{\text{occ}}(\mathbf{x} \mid \mathcal{G}) = 1 - \prod_{i=1}^N \big(1 - p_i(\mathbf{x})\big) \approx \sum_{i=1}^N p_i(\mathbf{x})$$

（当每个 $p_i(\mathbf{x}) \ll 1$ 时近似成立——对于空间中的任意点，只有少数附近的高斯球有显著贡献）

### 2.2 VGGT 观测的似然

VGGT-Ω 提供了每像素的深度观测。对于像素 $(u,v)$，有两种情况：

**情况 1：VGGT 观测到表面**（有深度值）。该像素对应的光线 $\mathbf{r}(t) = \mathbf{o} + t \cdot \mathbf{d}$，在深度 $d = D^{\text{vggt}}(u,v)$ 处有一个表面。

观测似然：
$$\mathcal{L}_{\text{surf}}(u,v) = p_{\text{occ}}(\mathbf{r}(d) \mid \mathcal{G}) = \sum_{i=1}^N o_i \cdot \mathcal{N}(\mathbf{r}(d); \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$

**情况 2：VGGT 观测到自由空间**（光线从相机到深度 $d$ 之间没有表面）。

观测似然（所有采样点都不被占据）：
$$\mathcal{L}_{\text{free}}(u,v) = \prod_{t < d} \big(1 - p_{\text{occ}}(\mathbf{r}(t) \mid \mathcal{G})\big) \approx \prod_{t < d} \big(1 - \sum_{i} o_i \mathcal{N}(\mathbf{r}(t); \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)\big)$$

### 2.3 负对数似然 → 损失函数

总负对数似然：
$$\mathcal{L} = -\sum_{\text{surf}} \log \mathcal{L}_{\text{surf}} - \sum_{\text{free}} \log \mathcal{L}_{\text{free}}$$

#### 表面项

$$-\log \mathcal{L}_{\text{surf}} = -\log \sum_{i=1}^N o_i \cdot \mathcal{N}(\mathbf{r}(d); \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$

这是 log-sum-exp 形式。使用 Jensen 不等式近似（最紧的下界是取最近的几个高斯）：

$$-\log \sum_i w_i \approx \min_i (-\log w_i) + C$$

但 $\min$ 不可微。使用 **soft-min**（温度 $\tau$）：
$$-\log \sum_i w_i \approx -\tau \log \sum_i w_i^{1/\tau}$$

当 $\tau \to 0^+$ 时收敛到 $\min$。

**实用近似——Chamfer Distance 视角**：

对 VGGT 表面点 $\mathbf{p} \in \mathcal{P}^{\text{vggt}}$，找到最近的高斯中心：
$$\mathcal{L}_{\text{chamfer}}(\mathcal{G}, \mathcal{P}^{\text{vggt}}) = \frac{1}{|\mathcal{P}|} \sum_{\mathbf{p} \in \mathcal{P}} \min_{i} \|\boldsymbol{\mu}_i - \mathbf{p}\|_2^2 + \frac{1}{N} \sum_{i=1}^N \min_{\mathbf{p} \in \mathcal{P}} \|\boldsymbol{\mu}_i - \mathbf{p}\|_2^2$$

第一项（forward）：每个 VGGT 表面点需要至少一个高斯靠近它。
第二项（backward）：每个高斯需要靠近某个 VGGT 表面点——**这惩罚了 floaters**。

这是 PM-Loss 验证过有效的形式，且**等价于将表面似然 $\mathcal{L}_{\text{surf}}$ 用最近邻近似后的负对数**。

#### 自由空间项

对每个自由空间光线，沿光线积分高斯密度：

$$\mathcal{L}_{\text{free}} = \sum_{\text{free}(u,v)} \sum_{t=0}^{d} \sum_{i=1}^N o_i \cdot \mathcal{N}(\mathbf{r}(t); \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) \cdot \Delta t$$

（利用 $-\log(1-x) \approx x$ for $x \ll 1$）

**换序求和的视角**：对每个高斯 $i$，沿所有包含它的自由光线积分：

$$\mathcal{L}_{\text{free}} = \sum_{i=1}^N o_i \cdot \underbrace{\sum_{\text{rays}} \int_{t < d_{\text{vggt}}} \mathcal{N}(\mathbf{r}(t); \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) dt}_{\equiv \Phi_i}$$

其中 $\Phi_i$ 是高斯 $i$ 被自由光线"穿透"的总概率质量。

#### 实用实现形式

直接沿光线采样的计算量太大。使用**逐高斯投影标记法**（POC 已验证）：

对每个高斯 $i$：
1. 投影到 VGGT 相机帧：$\tilde{\boldsymbol{\mu}}_i = \mathbf{T}_v^{-1} \cdot \boldsymbol{\mu}_i$（相机坐标）
2. 像素坐标：$(u_i, v_i) = \pi_{\mathbf{K}}(\tilde{\boldsymbol{\mu}}_i)$
3. 比较深度：$\Delta d_i = \tilde{\mu}_i^z - D^{\text{vggt}}(u_i, v_i)$
4. 自适应阈值：$\sigma_i = \kappa \cdot \max(\lambda_1(\boldsymbol{\Sigma}_i), \lambda_2(\boldsymbol{\Sigma}_i), \lambda_3(\boldsymbol{\Sigma}_i))$

标记规则：
$$y_i = \begin{cases}
1 & \text{if } |\Delta d_i| \leq \sigma_i \quad \text{(OCCUPIED — 靠近表面)}\\
0 & \text{if } \Delta d_i < -\sigma_i \quad \text{(FREE — 在表面前方)}\\
\text{mask} & \text{if } \Delta d_i > \sigma_i \quad \text{(UNKNOWN — 在表面后方，被遮挡)}
\end{cases}$$

### 2.4 最终损失函数

#### $\mathcal{L}_{\text{depth}}$ — 几何位置损失

对标记为 OCCUPIED 的高斯集 $\mathcal{O} = \{i \mid y_i = 1\}$：

$$\boxed{\mathcal{L}_{\text{depth}} = \frac{1}{|\mathcal{P}^{\text{vggt}}|} \sum_{\mathbf{p} \in \mathcal{P}^{\text{vggt}}} \min_{i \in \mathcal{O}} \|\boldsymbol{\mu}_i - \mathbf{p}\|_2^2 + \frac{1}{|\mathcal{O}|} \sum_{i \in \mathcal{O}} \min_{\mathbf{p} \in \mathcal{P}^{\text{vggt}}} \|\boldsymbol{\mu}_i - \mathbf{p}\|_2^2}$$

这是双向 Chamfer Distance。**梯度同时更新高斯位置 $\boldsymbol{\mu}_i$ 和编码器权重**（因为 $\boldsymbol{\mu}_i$ 来自 cost volume 的深度预测 + Gaussian adapter 的偏移量）。

#### $\mathcal{L}_{\text{occ}}$ — 占据分类损失

对标记集 $\mathcal{L} = \mathcal{O} \cup \mathcal{F}$（occupied + free）：

$$\boxed{\mathcal{L}_{\text{occ}} = -\frac{1}{|\mathcal{L}|} \sum_{i \in \mathcal{L}} \Big[ w_1 \cdot y_i \cdot \log o_i + w_0 \cdot (1 - y_i) \cdot \log(1 - o_i) \Big]}$$

类平衡权重（处理 3.6% occ vs 96.4% free 的不平衡）：
$$w_1 = \frac{|\mathcal{L}|}{2 \cdot |\mathcal{O}|}, \quad w_0 = \frac{|\mathcal{L}|}{2 \cdot |\mathcal{F}|}$$

可选增强：**Focal Loss**（$\gamma=2$）替代 BCE，自动降低易分类样本的梯度：
$$\mathcal{L}_{\text{occ}}^{\text{focal}} = -\frac{1}{|\mathcal{L}|} \sum_{i} \Big[y_i(1-o_i)^\gamma \log o_i + (1-y_i)o_i^\gamma \log(1-o_i)\Big]$$

#### $\mathcal{L}_{\text{free}}$ — 自由空间正则化

对标记为 FREE 的高斯集 $\mathcal{F} = \{i \mid y_i = 0\}$：

$$\boxed{\mathcal{L}_{\text{free}} = \frac{1}{|\mathcal{F}|} \sum_{i \in \mathcal{F}} \max(0, o_i - \epsilon)^2}$$

**为什么是 squared hinge 而非 BCE？**

自由空间中的高斯被标记为 $y_i=0$，但这是因为"光线没有被遮挡"，而非"高斯一定不在那里"。hinge loss 的含义是："低于阈值 $\epsilon$ 就够好，无需强制等于 0"——更鲁棒。

选择 $\epsilon = 0.05$（接近 0 但不强制等于 0）。

**数学依据**：
$$\mathcal{L}_{\text{free}} = \mathbb{E}_{\mathbf{x} \sim \text{free rays}} \left[\| \max(0, p_{\text{occ}}(\mathbf{x}) - \epsilon) \|_2^2 \right]$$

在逐高斯标记近似下退化为上述形式。

#### $\mathcal{L}_{\text{semantic}}$ — 语义分类损失

对 OCCUPIED 的高斯，使用 VGGT 像素投射的语义标签 $c_i$（来自 SAM2/YOLO）：

$$\boxed{\mathcal{L}_{\text{semantic}} = -\frac{1}{|\mathcal{O}|} \sum_{i \in \mathcal{O}} \log \frac{\exp(s_i^{c_i})}{\sum_{c=1}^K \exp(s_i^c)}}$$

标准交叉熵。但如果 VGGT+SAM2 的语义标签有噪声（如分割边界模糊），可以用 **label smoothing**（$\alpha=0.1$）。

#### $\mathcal{L}_{\text{color}}$ — 颜色渲染（辅助）

$$\boxed{\mathcal{L}_{\text{color}} = \lambda \cdot \frac{1}{HW} \sum_{u,v} \Big[ |I_{\text{rend}}(u,v) - I_{\text{gt}}(u,v)|_1 + \lambda_{\text{ssim}} \cdot (1 - \text{SSIM}(I_{\text{rend}}, I_{\text{gt}})) \Big]}$$

其中 $\lambda = 0.1$（辅助权重），$\lambda_{\text{ssim}} = 0.2$。

**为什么不能完全去掉 $\mathcal{L}_{\text{color}}$？**
- 颜色渲染信号保持 cost volume 的预训练知识不退化
- SSIM 项提供结构梯度，对遮挡/纹理弱区域有额外的约束作用
- 但权重必须小（0.1），否则优化会回归到 photometric-dominant 模式

### 2.5 总损失与训练策略

$$\boxed{\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{depth}} + \beta \cdot \mathcal{L}_{\text{occ}} + \gamma \cdot \mathcal{L}_{\text{free}} + \delta \cdot \mathcal{L}_{\text{semantic}} + \eta \cdot \mathcal{L}_{\text{color}}}$$

**分阶段训练**（防止梯度冲突）：

```
Stage 1 (Warmup, ~30% iterations):
  α=1.0, β=0.3, γ=0.1, δ=0, η=0.3
  目标：Gaussians 移动到 VGGT 表面附近

Stage 2 (Main, ~50% iterations):
  α=0.5, β=1.0, γ=0.5, δ=0.3, η=0.1
  目标：训练占据判断 + 自由空间感知 + 语义

Stage 3 (Fine-tune, ~20% iterations):
  α=0.3, β=1.0, γ=1.0, δ=0.5, η=0.05
  目标：精细化，颜色信号退场
```

---

## 三、RL 学习超参数组合

### 问题定义

我们有 **10+ 个连续/离散超参数** 需要调优：

| 超参数 | 类型 | 范围 | 含义 |
|--------|------|------|------|
| α, β, γ, δ, η | 连续 | [0, 2] | 损失权重 |
| ε | 连续 | [0.01, 0.2] | 自由空间阈值 |
| κ | 连续 | [1, 5] | 自适应 depth-threshold 倍数 |
| lr | 连续（log） | [1e-5, 1e-2] | 学习率 |
| λ_ssim | 连续 | [0, 0.5] | SSIM 权重 |
| τ_warmup | 离散 | [100, 500, 1000] | Stage 1 迭代数 |
| γ_focal | 连续 | [0, 5] | Focal loss γ |
| optimizer | 离散 | {Adam, AdamW, SGD} | 优化器选择 |

### RL 超参优化——现有方法

**RLGS**（arXiv 2025.08）已证明 RL 可以自适应调整 3DGS 训练超参数：
- 用轻量 policy 网络调整学习率和 densification 阈值
- +0.7 dB PSNR，无需额外计算开销

**但在我们场景中，RL 不是最合适的选择**：
- 超参搜索空间大（10+ dim），RL 的样本效率不够（每步需要一个 partial training run）
- RL 更适合**在线自适应调参**（训练过程中动态调整）而非**初始搜索**

### 推荐方案：Optuna + Population-Based Training (PBT)

| 阶段 | 方法 | 说明 |
|------|------|------|
| 初始搜索 | **Optuna (TPE sampler)** | 50-100 次 trial，每次 50 epochs 快速筛选 |
| 训练中自适应 | **PBT** (Population-Based Training) | 5-10 个 worker，周期性地 exploit+explore |
| 精细调参 | **Bayesian Optimization (GP)** | 对最佳候选区间的精细调优 |

**为什么 PBT 比 RL 更适合**：
1. PBT 直接优化验证集指标（BEV coverage），不需要设计 reward function
2. PBT 的 exploit（copy better weights）+ explore（perturb hyperparams）天然适合损失权重的动态调整
3. PBT 已在大规模训练中被验证（DeepMind 的 Capture the Flag, GAN 训练等）

### RL 的真正用武之地

**不是**全局超参搜索，而是**训练过程中的自适应高斯密度分配**：

RLD-GS（ICONIP 2025）的做法：
- 将每个区域的高斯密度分配建模为 MDP
- State：当前区域的 BEV coverage / VGGT depth error
- Action：增加/减少该区域的高斯数量
- Reward：BEV occupancy accuracy 提升

这可以在 VGGT 提供"困难区域"先验（深度估计置信度低的区域 → 需要更多 Gaussians）的指导下，让 RL 学习最优的高斯密度分配策略。

---

## 四、修订后的 Phase B 方案总结

| 组件 | 原方案 | 修订方案 | 理由 |
|------|--------|---------|------|
| Teacher | VGGT | **VGGT-Ω** (CVPR 2026 Oral) | 精度+26%, 推理1.6× |
| Student | MVSplat | **ReSplat** (优先) 或 CoSplat (备选) | 16× 更少高斯, recurrent refinement |
| 损失推导 | 直觉组合 | **概率模型 → NLL → 可计算近似** (见 §2) | 严谨可审稿 |
| 占据监督 | post-hoc labeling | **逐高斯投影 + 自适应阈值** ($\sigma_i = \kappa \cdot \max\text{eig}(\Sigma)$) | POC验证可行 |
| 自由空间 | ❌ 无 | **hinge loss** $\max(0, o_i - \epsilon)^2$ | 独有的创新 |
| 超参搜索 | 手动 | **Optuna → PBT → BO** 三阶段 | 自动化 |
| RL 应用 | — | **高斯密度自适应分配** (训练时, online) | RLD-GS 已验证 |

---

## 参考文献

1. Xu et al., "ReSplat: Learning Recurrent Gaussian Splatting," arXiv:2510.08575, 2025.
2. CoSplat authors, "CoSplat: Resolving multi-view inconsistency in feed-forward 3DGS," PR, 2026.
3. Wang et al., "VGGT-Ω," CVPR 2026 Oral, arXiv:2605.15195.
4. Shi et al., "PM-Loss: Revisiting Depth Representations for Feed-Forward 3DGS," 3DV 2026.
5. Gan et al., "GaussianOcc," ICCV 2025.
6. Li et al., "RLGS: RL-Based Adaptive Hyperparameter Tuning for GS," arXiv:2508.04078.
7. Huang et al., "RLD-GS: RL-Driven Gaussian Splatting," ICONIP 2025.
8. Jaderberg et al., "Population Based Training of Neural Networks," arXiv:1711.09846.
9. Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework," KDD 2019.
