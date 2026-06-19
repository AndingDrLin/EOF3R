# Phase B 训练显存估算 (A6000 48GB)

> 日期：2026-06-19
> 目标：估算 ReSplat + VGGT 跨模型几何蒸馏训练在 A6000 上是否会 OOM

---

## 1. 模型参数量

| 组件 | 参数量 | Float32 | Float16 (AMP) |
|------|--------|---------|---------------|
| ReSplat-base encoder | 223M | 892 MB | 446 MB |
| UniMatch depth backbone | ~86M | 344 MB | 172 MB |
| Point Transformer (6 blocks) | ~30M | 120 MB | 60 MB |
| OccupancyHead | ~100K | 0.4 MB | 0.2 MB |
| SemanticHead (10 classes) | ~100K | 0.4 MB | 0.2 MB |
| **总计** | **~340M** | **~1.36 GB** | **~0.68 GB** |

## 2. 训练状态显存

| 组件 | 公式 | 显存 |
|------|------|------|
| 模型参数 (AMP) | 340M × 2 bytes | 0.68 GB |
| 梯度 (float32) | 340M × 4 bytes | 1.36 GB |
| AdamW 优化器状态 | 2 × 340M × 4 bytes | 2.72 GB |
| **训练状态总计** | | **~4.76 GB** |

## 3. 激活值显存 (主要瓶颈)

### 3.1 UniMatch/ViT-Base backbone
- 输入: (B, 2, 3, 256, 256) → patch embedding → (B, 2, 256, 16×16)
- 中间激活: 12 layers × 768 dim × 256 tokens × B
- 估算: **~2-3 GB** (AMP, batch_size=1)

### 3.2 Cost Volume
- 128 depth candidates × 64×64 spatial × 128 channels
- 128 × 64 × 64 × 128 × 4 bytes = **256 MB**

### 3.3 Point Transformer (KNN attention)
- 6 blocks, 16 neighbors, 64 channels
- KNN 索引 + attention 权重 + 中间特征
- 估算: **~1-2 GB** (这是最不确定的部分)

### 3.4 gsplat 渲染器
- 前向渲染 + 反向传播需要的中间状态
- 估算: **~0.5-1 GB**

### 3.5 激活值总计
- **~4-7 GB** (batch_size=1, AMP)

## 4. 输入数据

| 组件 | 大小 | 显存 |
|------|------|------|
| 输入图像 (B=1, V=2, 3, 256, 256) | AMP | 0.75 MB |
| 相机参数 | | ~1 KB |
| VGGT 深度图 (B=1, H, W) | | ~0.5 MB |
| VGGT 表面点云 (500 points) | | ~6 KB |
| **总计** | | **~2 MB** |

## 5. 总显存估算

### batch_size=1 (Phase B 训练)

| 组件 | 显存 |
|------|------|
| 训练状态 (模型+梯度+优化器) | 4.76 GB |
| 激活值 | 4-7 GB |
| 输入数据 | 0.002 GB |
| **总计** | **~9-12 GB** |

### batch_size=4

| 组件 | 显存 |
|------|------|
| 训练状态 | 4.76 GB |
| 激活值 (×4) | 16-28 GB |
| 输入数据 (×4) | 0.008 GB |
| **总计** | **~21-33 GB** |

### batch_size=8

| 组件 | 显存 |
|------|------|
| 训练状态 | 4.76 GB |
| 激活值 (×8) | 32-56 GB |
| 输入数据 (×8) | 0.016 GB |
| **总计** | **~37-61 GB** ⚠️ |

## 6. 结论

### ✅ A6000 (48GB) 安全配置

| 配置 | 预估显存 | 推荐度 |
|------|----------|--------|
| batch_size=1, AMP | 9-12 GB | ✅ 安全 |
| batch_size=2, AMP | 13-19 GB | ✅ 安全 |
| batch_size=4, AMP | 21-33 GB | ✅ 安全 |
| batch_size=6, AMP | 29-47 GB | ⚠️ 接近上限 |

### ⚠️ 风险点

1. **Point Transformer KNN attention** — 显存估算不确定，可能比预期高
2. **gsplat 反向传播** — 需要存储每个 Gaussian 的中间状态
3. **VGGT 在线推理** — 如果同时加载 VGGT，额外 ~6 GB

### 💡 优化建议

1. **使用 AMP (Automatic Mixed Precision)** — ReSplat 默认开启，可节省 ~40% 激活值显存
2. **Gradient Checkpointing** — ReSplat 配置支持 `use_checkpointing: true`，可减少 ~50% 激活值，但训练慢 ~30%
3. **预计算 VGGT 监督** — 不要在训练时同时运行 VGGT，离线预计算后读取
4. **从 batch_size=1 开始** — 先验证流程，再逐步增大

### 推荐训练配置

```yaml
# Phase B 训练配置 (A6000 48GB)
phase_b:
  training:
    batch_size: 4           # 安全起见，从 4 开始
    use_amp: true           # 必须开启
    gradient_checkpointing: false  # 如果 OOM 再开
    gradient_accumulation: 2  # 有效 batch_size = 8
  student:
    model: resplat-base     # 223M params
    num_refine: 0           # 训练时不用 refinement
```

### 原始 ReSplat 参考

- 原始 ReSplat 在 A100 (80GB) 上训练 batch_size=14
- A6000 (48GB) 约为 A100 的 60% 显存
- 预估最大 batch_size = 14 × 0.6 ≈ 8（但有额外的 occupancy/semantic heads，建议保守用 4-6）
