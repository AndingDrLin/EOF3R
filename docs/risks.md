# 风险评估

> 每个风险包含：概率（H/M/L）、影响（H/M/L）、缓解措施、降级方案。
> 降级方案的意义是：即使该风险完全发生，仍然有可接受的论文产出。

---

## 1. 科研风险

### R1: Feedforward Gaussian Occupancy 几何精度不够

- **概率**: M | **影响**: H
- **说明**: G2O-inspired feedforward Gaussian occupancy 在少视角（2-4 个 crop）输入下，物体 Gaussian 的几何精度可能不足以生成有用的 BEV footprint。模型可能学会降低 occupancy_alpha 来掩盖几何不确定性，导致 BEV 投影后的 footprint 过小或空洞。
- **缓解**: 优先在 ScanNet++ 上离线评估 occupancy accuracy（Chamfer Distance, F-Score, Footprint IoU），确认可接受后再接入机器人 pipeline。G2O 几何约束（geometry scaffold, edge supervision）本身就是缓解措施。
- **降级**: 回退到 per-object 优化式 3DGS（仍然比全场景训练快，只需要一个物体的少量迭代）。如果还不够，退到直接用 point cloud / depth 估计生成 BEV footprint（更粗但更可靠）。

### R2: BEV 代价地图增强效果不明显

- **概率**: M | **影响**: H
- **说明**: 如果物体形状估计误差大，或语义风险分级的权重设置不合理，云端增强 costmap 可能和纯 LiDAR obstacle layer 差不多。这是论文核心贡献的风险。
- **缓解**: 预先设计好消融实验梯度（纯 LiDAR → LiDAR + 粗物体形状 → LiDAR + 精确形状 → + 语义风险），确保至少有一个梯度可以看到差异。多场景多次运行取统计显著性。
- **降级**: 如果全部条件下差异都不显著，论文贡献变为"对云端增强何时有效的表征分析"，诚实报告 null result 也是科研成果。同时加强占据几何方向的定量评估（Chamfer Distance, F-Score, Footprint IoU），确保论文有其他支撑。

---

## 2. 工程风险

### R3: 系统集成复杂度超标

- **概率**: H | **影响**: M
- **说明**: 项目涉及 SAM2 + 3DGS/MVSplat + VGGT/MASt3R + Nav2 + ROS2 + 车云通信，模块数量多，依赖关系复杂。如果每个模块都有独立的 conda 环境和版本要求，集成调试将非常耗时。
- **缓解**:
  - 每个模块独立可运行、独立可测试，不依赖其他模块启动。
  - Phase 1（离线验证）完全不需要 ROS2 和通信模块，大幅降低早期复杂度。
  - 模块接口用文件/标准格式（.ply, .png, .yaml, .json）传递，不用内存/网络传递。
- **降级**: 如果 Phase 2-3 集成困难，论文基于 Phase 1（离线 pipeline）完成。Phase 2-3 作为"系统设计"章节讨论。

---

## 3. 算力风险

### R4: GPU 显存不足

- **概率**: H | **影响**: M
- **说明**: VGGT + 3DGS + SAM2 同时对 GPU 显存压力大。目标硬件 RTX 3060/4060（12GB VRAM）可能不够同时加载多个模型。
- **缓解**:
  - 所有模型串行执行，上一个完成后 `del` + `torch.cuda.empty_cache()`。
  - 使用小模型变体（SAM2 tiny、VGGT 低分辨率、减少 3DGS 高斯球数量）。
  - 优先处理 ROI 内的物体，远处物体用粗模型。
- **降级**: 在云端服务器（更高显存 GPU）上运行完整 pipeline。车端只做推理结果的可视化和融合。如果连服务器显存也不够，减少同时处理的物体数量。

---

## 4. 通信延迟风险

### R5: 云端推理延迟过高

- **概率**: H | **影响**: M
- **说明**: 校园 WiFi 环境下，关键帧上传 + 云端推理 + costmap 下载的端到端延迟可能超过 5 秒。如果延迟 >3s，机器人已经移动了 1.5m（在 0.5 m/s 下），云端结果基本无用。
- **缓解**:
  - 关键帧选择策略：只上传每 N 帧中的 1 帧（N=5-10），减少上传量。
  - 图像压缩（JPEG 质量 80），不传 raw。
  - 云端渐进式返回：先返回粗 costmap（<1s），再返回精细 costmap（<5s）。
  - 车端 timeout 设为 3s，超时直接丢弃。
- **降级**: 云端结果只用于离线可视化和低频 costmap 增强（每 10 秒更新一次）。Phase 1 完全离线，不受此影响。Phase 2 用有线网络或本地 WiFi 热点降低延迟。

---

## 5. 数据风险

### R6: 缺少校园机器人场景数据

- **概率**: M | **影响**: H
- **说明**: 截至项目开始，没有现成的 Husky 在校园/园区的配送场景数据（rosbag + 标定参数）。需要自行采集，可能受天气、场地、设备可用性的限制。
- **缓解**:
  - 先使用 ScanNet++ 室内数据验证核心重建 pipeline（前景-背景分解 → 3DGS → costmap）。
  - 在 Gazebo / Isaac Sim 中搭建仿真场景，生成合成 rosbag 用于系统集成测试。
  - 尽早预约 Husky 和校园场地，录制至少 3 个场景的 rosbag。
- **降级**: 如果真实 Husky 数据无法采集，使用仿真数据完成 pipeline 验证和 costmap 对比。论文中讨论仿真到真实的迁移挑战作为 future work。

---

## 6. 安全风险

### R7: 真实机器人测试安全

- **概率**: L | **影响**: H
- **说明**: Husky 是中型无人车（~50kg），在 0.5 m/s 速度下仍然可能造成碰撞伤害。
- **缓解**:
  - Phase-gated：Phase 1 全部离线，不涉及真实机器人运动。
  - Phase 2 机器人只采集数据，不做自动控制。
  - Phase 3 限速 0.2 m/s，空旷场地测试，人类操作员始终手持物理急停。
  - ROS2 safety controller 硬限速 0.5 m/s，心跳丢失自动停车。
- **降级**: Phase 3 不做真实闭环。改为 rosbag replay + RViz 可视化 + 离线路径规划对比。论文中有完整系统设计但实验在离线/半在线条件下完成。

---

## 7. 时间风险

### R8: 毕设周期内无法完成所有阶段

- **概率**: H | **影响**: H
- **说明**: 8 阶段路线 + 三阶段工程实现的完整范围对于一个本科毕设（约 4-6 个月实际工作时间）可能过于庞大。
- **缓解**:
  - 明确定义 MVP：Phase 1（离线验证）+ Phase 2（半在线 RViz 显示）= 论文及格线。
  - Phase 3（低速闭环）是 stretch goal，不依赖它完成论文。
  - 每个阶段的产出独立有价值，不要求全部完成才能写论文。
  - 如果时间紧张，砍掉 communication 模块，Phase 2 用"手动拷贝文件模拟云端返回"的方式演示。
- **降级**: 论文基于 Phase 1 完成。Phase 2-3 作为"系统设计"和"future work"。

---

## 8. 模型效果风险

### R9: Photometric-Occupancy Misalignment

- **概率**: M | **影响**: M
- **说明**: 如果模型训练中 RGB photometric loss 和 occupancy loss 的目标不完全一致，可能出现 PSNR 尚可但 occupancy_alpha 质量差的情况（模型学会用透明度"作弊"来降低 RGB loss，而非学习正确的几何占据）。这是将 photorealistic 模型改造为 planning-oriented 模型的核心风险。
- **缓解**:
  - RGB loss 权重设置为 0.1（辅助监督），梯度主要通过 occupancy/mask/silhouette loss 回传。
  - 训练过程中同时监控 RGB 指标和 occupancy 指标，确认两者不出现严重背离。
  - 消融实验：对比 L_rgb=0.1（推荐）vs L_rgb=1.0（等权重）vs L_rgb=0（纯 occupancy），验证 RGB loss 作为辅助监督是否真的有益。
- **降级**: 如果 RGB loss 干扰 occupancy 学习，将其权重降为 0（纯几何监督训练）。论文中诚实分析 photometric 和 occupancy 训练目标的张力。

### R10: Occupancy Boundary Artifacts

- **概率**: M | **影响**: M
- **说明**: BEV 投影时，occupancy_alpha 阈值的选择直接影响 footprint 形状。阈值过高 → footprint 过小（漏掉占据区域）；阈值过低 → footprint 过大（和 LiDAR 膨胀一样保守，失去增强意义）。不同物体的最优阈值可能不同。
- **缓解**:
  - 在验证集上 grid search occupancy_alpha 阈值（0.1-0.9），选择 Footprint IoU 最优的值。
  - 引入 boundary-aware edge loss（G2O-inspired），显式监督 footprint 边界质量。
  - 输出 confidence/uncertainty，允许 costmap 融合时根据 confidence 调整 alpha 阈值。
- **降级**: 使用固定的保守阈值（α > 0.3），宁可稍微过大也不漏掉占据。论文中报告不同阈值下的 footprint IoU 曲线。

### R11: 个别模块效果不达预期

- **概率**: M | **影响**: L
- **说明**: SAM2 在某些光照/角度下分割错误、VGGT 在少纹理区域估计失败、Gaussian occupancy 对透明/反光物体质量差等。
- **缓解**:
  - 每个模块都有 fallback：SAM2 失败 → YOLO bbox detection；VGGT 失败 → DUSt3R；Gaussian occupancy 失败 → point cloud footprint。
  - 在数据筛选阶段就排除极端困难的场景（强逆光、大面积镜面反射）。
- **降级**: 选择"容易"场景完成 pipeline 验证。论文中诚实地讨论失败案例和适用范围。

---

## 降级路径总览

```
Gaussian occupancy 精度不够  →  per-object 优化 3DGS  →  point cloud footprint
Photometric-occupancy 背离   →  纯几何监督 (L_rgb=0)  →  诚实分析训练张力
Occupancy boundary 不准      →  保守 alpha 阈值       →  confidence 自适应阈值
3R 模型太重                  →  轻量 depth estimation →  单目 depth + 地面假设
SAM2 太慢                    →  detector + tracking   →  只在关键帧用 SAM2
云端延迟太高                 →  离线处理 + RViz 回放  →  低频增强(10s/次)
真实 Husky 不可用             →  Gazebo 仿真           →  纯 rosbag 离线
闭环测试不安全               →  半在线 RViz            →  离线规划对比
系统太复杂                   →  Phase 1 only          →  各模块独立 demo
效果不明显                   →  诚实报告 + 消融分析   →  转为"局限性研究"
```

**关键原则**：每一步降级都产出可展示、可评估、可写进论文的结果。不存在"全有或全无"的阻塞点。
