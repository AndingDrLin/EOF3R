# 项目范围与目标

## 标题

Efficient Object-level Feedforward 3D Reconstruction with 3D Gaussian Splatting

**副标题**：Planning-Oriented Gaussian Occupancy for Low-Speed Campus Robot Navigation
**不追求逼真重建，而是面向规划的几何-语义表示。**

本科毕业设计原型系统。

---

## 要解决的问题

低速无人车（Husky）在校园/园区执行"最后 50 米"配送任务时，会遇到自行车、电动车、纸箱、快递箱、书包、路锥、倒地自行车、临时围挡、行人低速横穿、窄通道、路沿、台阶等复杂近场障碍物。

本地避障（LiDAR + 超声波 + Nav2 obstacle layer）可以安全停车或绕开，但存在以下问题：

- **过度保守**：纯几何避障将所有未知物体当作刚性墙壁，导致不必要的停车或大角度绕行。
- **形状信息缺失**：点云稀疏时，障碍物实际占据形状估计不准，costmap 可能过大或过小。
- **语义信息缺失**：所有障碍物被同等对待（都是"不可通行"），无法区分"可以靠近的路锥"和"必须远离的行人"。
- **路径质量差**：几何噪声和过度保守导致路径抖动、反复停顿、绕行不合理。

**核心 idea**：融合 G2O 几何约束思想，用 feedforward 方式从 RGB、LiDAR / depth、odometry 等多模态车载观测中端到端预测机器人可用的结构化世界状态。系统仍采用 **object-level、前景与背景分开表示** 的路线：前景物体以 Gaussian occupancy 表达占据形状、语义、风险、置信度和 BEV footprint；背景以 3R / VGGT-like 前馈模型估计粗几何、free-space、unknown-space、occlusion boundary 和可通行区域；二者再融合为 BEV occupancy / semantic costmap，作为 Nav2、世界模型、VLA 或 agent 的输入。系统不追求逼真渲染或传统意义上的完整三维重建，而追求更可靠、更省资源、更适合规划和智能体决策的几何-语义世界表征。

---

## 做的（scope 内）

1. **前景物体 G2O-inspired feedforward Gaussian occupancy 预测**：从 RGB、LiDAR / depth、odometry、few-shot object crops 和 masks 中，单次前向传播预测 object-level geometry-semantic Gaussian primitives（3D center, size, orientation, occupancy_alpha, BEV footprint, semantic class, risk score, confidence, uncertainty）。RGB/color 为辅助输出，SH/view-dependent appearance 不是核心目标。
2. **背景 3R / VGGT-like 前馈世界表征估计**：用 DUSt3R / MASt3R / VGGT-like 模型从场景图像、LiDAR / depth 和位姿信息中估计粗深度、pointmap、相机关系、地面结构、free-space、unknown-space、occlusion boundary 和可通行区域。目标是为 agent / world model / planner 提供背景状态描述，不是逼真背景渲染。
3. **Planning-oriented 训练目标设计**：以 L_occupancy + L_mask + L_depth + L_silhouette + L_footprint + L_semantic + L_confidence 为核心损失函数，RGB photometric loss 仅作为辅助监督（权重 ≤0.1）。训练目标不主要围绕渲染质量，而围绕占据精度、边界精度和语义正确性。
4. **语义融合与 BEV 代价地图生成**：前景物体 Gaussian occupancy 投影到 BEV 占据网格（使用 occupancy_alpha 阈值 + confidence 加权），结合语义信息（类别、实例 ID、风险等级、可通行属性），生成 Nav2 兼容的 costmap layer。
5. **车端-云端异步架构**：车端始终运行本地安全回路（相机采集、里程计、IMU、本地 costmap、急停、Nav2 局部规划、cmd_vel 控制），云端负责高算力推理（SAM2 mask refinement、3R 背景几何估计、G2O-inspired feedforward Gaussian occupancy、语义融合、costmap patch 生成）。云端返回 lightweight planning-oriented representation（object state, 3D bbox, BEV footprint, semantic label, risk score, confidence, costmap patch），不返回完整 Gaussian 渲染模型。云端结果作为异步增强，不参与车辆实时控制。
6. **导航质量对比实验**：在 3 个近场场景中，对比纯本地避障 vs 云端增强规划的路径质量差异（路径平滑度、不必要停车次数、绕行比例等）。
7. **系统降级验证**：验证云端延迟过大时自动丢弃、通信中断时降级为本地避障、恢复通信后重新增强的能力。

---

## 不做的（非 scope）

1. **不做完整自动驾驶**：不涉及开放道路、高速行驶（>1 m/s）、交通规则、路口决策。
2. **不做实时在线 SLAM**：不依赖实时定位建图，场景重建是异步的。
3. **不做 dynamic scene / 4D 重建**：只处理静态或准静态场景（行人/自行车在低速互动场景中出现，但跟踪和预测由本地安全回路处理，不依赖云端）。
4. **不做 open-vocabulary 检测**：语义信息来自预定义的类别集合（人、自行车、路锥、纸箱等），不做任意物体检测。
5. **不发明新 3DGS 变体、新 feedforward 架构**：使用现有开源模型（MVSplat、3DGS、VGGT、DUSt3R、MASt3R、SAM2）。
6. **不做云端直接车辆控制**：云端输出是 costmap 增强层，不是 cmd_vel。本地安全回路始终独立运行。
7. **不做高速导航**：最大测试速度 0.5 m/s，始终有物理急停开关。
8. **不做全局路径规划**：只增强 Nav2 局部规划器，全局规划使用标准 Nav2 global planner。
9. **不做动态障碍物跟踪与预测**：这是本地安全回路和未来工作的内容。
10. **不追求 photorealistic reconstruction / novel view synthesis**：不把 PSNR/SSIM/LPIPS 作为核心指标，不把高阶 SH、view-dependent color appearance 作为核心输出，不把逼真渲染作为项目目标。

---

## 研究 vs 工程

| 研究 | 工程集成 |
|------|----------|
| G2O-inspired feedforward Gaussian occupancy 方法设计 | 调用 SAM2 做分割 |
| 非 photorealistic 训练目标设计（occupancy/mask/silhouette/depth loss） | 调用 3DGS / MVSplat 官方实现 |
| Occupancy_alpha 驱动的 BEV footprint 投影方法 | 调用 VGGT / DUSt3R / MASt3R 官方模型 |
| 语义代价地图对局部规划质量的影响（消融实验） | ROS2 Nav2 costmap layer plugin 开发 |
| 车端-云端异步感知的延迟-收益 trade-off | 车端-云端通信模块开发 |
| | 安全降级逻辑实现 |
| | 3 个导航实验场景搭建与运行 |

---

## 成功标准

### 最低标准（必须达到）

1. 完整 pipeline 在至少 3 个场景的离线数据上端到端运行（分割 → 前景 Gaussian occupancy 预测 → 背景几何估计 → costmap 生成 → 路径规划可视化）
2. 云端增强 costmap 在至少 2 个场景中产生可见的路径质量改善（与纯本地避障对比）
3. 系统降级行为正确：云端延迟/中断时车端继续运行不崩溃
4. 物体 occupancy 精度可量化评估（Chamfer Distance, F-Score, Footprint IoU vs GT）

### 期望标准

5. 路径平滑度（integrated curvature）与纯本地避障相比有统计显著改善
6. 不必要停车次数减少 ≥30%（在窄通道场景中）
7. 物体 BEV footprint 估计误差 <15 cm（与实测量测对比）
8. Phase 2（半在线 RViz 显示）完整可运行

### 延伸标准（不强制）

8. Phase 3（低速闭环 demo）在真实 Husky 上完成至少 1 个场景
9. 消融实验证明语义信息对规划质量的独立贡献

---

## 系统安全声明

以下声明是系统设计的硬约束，不可妥协：

1. **本地安全回路始终独立运行**。车端的急停、速度限制、Nav2 局部规划不依赖云端。
2. **云端结果是异步增强，不是控制指令**。云端生成 costmap patch，车端自行决定是否融合到局部 costmap 中。
3. **最大测试速度硬限制为 0.5 m/s**。由 ROS2 safety controller 强制实施。
4. **任何时候人类操作员都持有物理急停开关**。
5. **通信中断或云端延迟 >3s 时，系统自动丢弃云端结果，降级为本地避障模式**。
