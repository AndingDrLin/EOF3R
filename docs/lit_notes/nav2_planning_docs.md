# Nav2 Local Planning: Controller Plugins (DWB, TEB, RPP, MPPI) — Architecture Overview

- **作者**：ROS2 Navigation Working Group (Open Robotics / Community)
- **会议/期刊**：Nav2 官方文档
- **年份**：持续更新
- **链接**：[docs.nav2.org](https://docs.nav2.org/) | [github.com/ros-planning/navigation2](https://github.com/ros-planning/navigation2)
- **阅读日期**：2025-06-16

## 一句话总结

Nav2 提供四种局部规划器插件（DWB/TEB/RPP/MPPI），其中 DWB 通过采样速度候选并综合多个 critic plugin 的评分选择最优轨迹，直接消费 local costmap；RPP 是简化纯跟踪器；所有规划器都要求 costmap 以 ≥5Hz 频率更新。

## 核心架构

1. **DWB (Dynamic Window Approach, 默认推荐)**：采样速度候选 (vx, vy, vtheta) → 模拟短时轨迹 → 每个 critic 打分 → 加权综合 → 选最优。Critic 插件包括：
   - BaseObstacleCritic：沿轨迹采样 costmap 值求和（障碍物代价）
   - ObstacleFootprintCritic：投影机器人 footprint 检查碰撞
   - PathAlign/GoalAlign：全局路径跟随
   - OscillationCritic：防止前后抖动
   - PreferForwardCritic：优先前进
2. **RPP (Regulated Pure Pursuit)**：自适应纯跟踪变体，比 DWB 简单。
3. **TEB (Timed Elastic Band)**：MPC 类优化，适用于 Ackermann/差速/全向底盘。维护状态不稳定。
4. **MPPI (Model Predictive Path Integral)**：MPC 变体，模块化代价函数，TEB 的推荐替代。
5. **Costmap 消费方式**：DWB 订阅 local costmap（典型 3m x 3m 滚动窗口 @ 5-10 Hz）。参数 `costmap_update_timeout: 0.3s`——超时后控制器阻塞。

## 关键要点

- **costmap 必须 ≥5 Hz 更新**，这是硬实时要求。云端异步增强（延迟 200+ms）不可能直接提供 5 Hz —— 必须作为持久化 costmap patch 融合到本地层。
- **BaseObstacleCritic 沿轨迹采样**意味着 costmap 中障碍物的**形状准确性**直接影响轨迹评分。如果云端提供的物体 footprint 更准确（vs 纯 LiDAR 的稀疏点云），DWB 的轨迹评分就会更准确。
- **inflation 参数因物体类别而异**：对行人大半径膨胀（安全距离），对路锥小半径膨胀（可靠近）。

## 与本文的关系

DWB 是本项目的"下游消费者"。所有云端计算（分割→前景占据→背景几何→融合→costmap）最终都要通过 DWB 的 BaseObstacleCritic 影响轨迹选择。对本项目的关键约束：(1) costmap 必须 ≥5 Hz 更新 → 云端不能是唯一来源；(2) 障碍物形状精度直接影响 DWB 评分 → 本项目的 Gaussian occupancy 比纯 LiDAR 点云提供更好的形状信息。

## 可用性

- [x] 文档完善
- [x] 代码开源
- [x] 可本地部署
- [x] N/A（非 ML 模型）

## 笔记

- DWB 的 critic-based 架构是本项目的天然集成点：本地 obstacle_layer 提供 5 Hz 安全保证（从 LiDAR 点云），云端 semantic layer 提供异步增强（更准确形状 + 语义标签 → 更智能的 inflation）。
- 本项目的关键设计洞见：cloud costmap patch 应该用 `updateWithMax()` 策略叠加到本地层——云端只增加 cost（更保守/更安全），不减少 cost（避免漏掉障碍物）。
