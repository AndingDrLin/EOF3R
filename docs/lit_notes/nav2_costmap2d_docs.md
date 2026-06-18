# Nav2 costmap_2d: Layered Costmap Architecture

- **作者**：ROS2 Navigation Working Group (Open Robotics / Community)
- **会议/期刊**：ROS2 Nav2 官方文档
- **年份**：持续更新 (Humble / Iron / Jazzy / Rolling)
- **链接**：[docs.nav2.org](https://docs.nav2.org/) | [Navigation Plugins](https://docs.nav2.org/plugins/)
- **代码**：[github.com/ros-planning/navigation2](https://github.com/ros-planning/navigation2)
- **阅读日期**：2025-06-16

## 一句话总结

Nav2 使用 Layered Costmap 架构（Lu et al., IROS 2014）：多个 costmap layer plugin 按序叠加，每层独立更新各自的 cost value，通过 updateWithMax/Addition/Overwrite 合并到 master grid，最后通常由 inflation layer 扩展障碍物代价。

## 核心架构

1. **LayeredCostmap**：管理 master costmap (Costmap2D) + 插件列表 + 滤波器列表。按序调用每层的 updateBounds() 和 updateCosts()，将各层 cost 以预设策略合并。
2. **Layer Plugin Interface**（nav2_costmap_2d::Layer）：所有层继承此抽象类，必须重写 updateBounds()、updateCosts()、reset()、isClearable()。
3. **内置 Plugins**：
   - StaticLayer: 静态地图（来自 SLAM）
   - ObstacleLayer: 2D 概率模型处理传感器数据
   - VoxelLayer: 3D voxel grid with raycasting
   - InflationLayer: 膨胀障碍物（exponential decay, OpenMP 加速）
   - SpatioTemporalVoxelLayer: 带时域衰减的 3D voxel
4. **Costmap Filters**：特殊类层，应用空间相关的 filter mask——KeepoutFilter（禁行区）、SpeedFilter（限速区）等。

## 关键要点

- **插件顺序决定数据流**：如果 InflationLayer 在 ObstacleLayer 之前，后续添加的障碍物不会被膨胀。
- **四种合并策略**：updateWithMax()（取最大值）、updateWithAddition()（叠加）、updateWithOverwrite()（覆盖）、updateWithTrueOverwrite()（完全覆盖）。
- **自定义 Plugin 开发**：继承 Layer → 重写必要方法 → PLUGINLIB_EXPORT_CLASS 导出 → 注册到 YAML config。

## 与本文的关系

**本项目 costmap 模块的直接目标**。我们需要开发一个自定义 Nav2 costmap layer plugin，将云端生成的 BEV semantic costmap 作为增强层叠加到本地 LiDAR obstacle layer 上。合并策略用 updateWithMax()（云端增强层只增加 cost，不减少本地层的安全保守估计）。云端结果超时时自动失活该层。

## 可用性

- [x] 文档完善
- [x] 代码开源
- [x] 可本地部署 (ROS2 apt)
- [x] 显存要求（不适用于此）

## 笔记

- Layered costmap 的设计天然支持本项目的"异步增强"架构——云端层作为一个独立的 Layer plugin，可以随时 activate/deactivate，超时后自动失活（不影响本地层）。
- inflation 参数（inflation_radius, cost_scaling_factor）需要根据语义类别动态调整：行人需要大的膨胀半径，路锥只需要小的膨胀半径。这可以通过在 cloud layer 中自带 inflation 实现。
- KeepoutFilter 和 SpeedFilter 的 filter 机制可以作为 Phase 2 的扩展——例如云端检测到高风险区域时下发 keepout filter mask 而非完整 costmap。
