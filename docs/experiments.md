# 导航实验设计

> 3 个实验场景，每个验证云端几何-语义增强的不同方面。
> 所有实验先在 Phase 1（离线 rosbag replay）运行，Phase 2（半在线 RViz）验证实时性，Phase 3（闭环）选做。

---

## 实验原则

1. **Baseline 是纯本地避障**（Nav2 local planner + LiDAR obstacle layer），不是"没有避障"。
2. **对比的是路径质量**，不是安全性（两者都应该安全）。
3. **每个实验一个独立假设**，不混在一起测试。
4. **至少 5 次重复**，交替运行 baseline 和 enhanced，消除环境变化的影响。
5. **所有实验数据公开可复现**（rosbag + config 文件 + 评估脚本）。

---

## 实验 1：窄通道中的不规则障碍物

### 场景布置

- **空间**：宽 2-3m 的通道（模拟宿舍楼/教学楼走廊或两排自行车之间的窄路）
- **障碍物**：
  - 自行车斜放（与通道成 45°，占据宽度的 60%）
  - 电动车侧倒（占据宽度的 50%，形状不规则）
  - 纸箱堆叠（大小不一，堆高 30-50cm）
  - 路锥 1 个
- **机器人任务**：从通道一端走到另一端，不碰撞任何障碍物
- **总路径长度**：约 8-10m

### 假设

- **H1**：纯 LiDAR 避障会过度估计障碍物占据区域（点云稀疏 + 膨胀），导致机器人停车或绕行远大于实际需要的距离。
- **H2**：云端 object-level 3DGS 提供更精确的物体形状 → BEV footprint 更接近实际占据 → 局部规划器可以找到更紧的通过路径。
- **H3**：语义信息（知道哪些物体可以靠得更近，如路锥 vs 电动车）进一步改善路径。

### 指标

| 指标 | 定义 | 预期方向 |
|------|------|----------|
| Path length | 机器人实际走过的路径总长度 (m) | Enhanced ≤ Baseline |
| Unnecessary stop count | 速度为 0 但前方无实际碰撞风险的次数 | Enhanced < Baseline |
| Minimum clearance | 路径上离障碍物最近的距离 (m) | 两者都 > 安全阈值 (0.3m) |
| Path smoothness | 路径曲率的积分 (rad/m) | Enhanced < Baseline |
| Time to goal | 从起点到终点的时间 (s) | Enhanced ≤ Baseline |
| Success rate | 无碰撞/无人工干预完成的次数 / 总次数 | 两者都应为 100% |

### 采集数据

- Husky RGB-D 相机图像（15 fps）
- 2D LiDAR 扫描
- 里程计 + IMU
- 机器人 cmd_vel
- 障碍物实物照片 + 实测尺寸（作为 footprint GT）
- 完整 rosbag

### 可视化产出

- BEV 视图：LiDAR 点云 vs 云端 3DGS footprint 叠加在同一帧
- 路径对比图：baseline 路径 vs enhanced 路径画在同一张 BEV 图上
- 物体形状对比：单个障碍物的实测轮廓 vs Gaussian BEV 投影轮廓
- 局部 costmap 快照：baseline 和 enhanced 的关键帧对比

### 统计

- 配对 t-test 或 Wilcoxon signed-rank test（取决于正态性检验结果）
- 报告 mean ± std，效应量（Cohen's d）

---

## 实验 2：低矮/不规则障碍物

### 场景布置

- **空间**：开放区域（模拟快递点门口或教学楼入口）
- **障碍物**：
  - 倒地自行车（高度 <30cm，形状扁平不规则）
  - 低障碍栏/临时围挡（高度 20-40cm，宽度大但薄）
  - 压扁纸箱（高度 <15cm，面积大）
  - 路沿/台阶（高度 5-15cm）
- **机器人任务**：穿过该区域到达指定目标点

### 假设

- **H1**：2D LiDAR 扫描平面在固定高度（Husky 通常 ~20cm），低矮障碍物可能被扫描到但点云稀疏，或者低于扫描平面完全漏掉。
- **H2**：RGB-D 相机可以看到这些障碍物，云端 3R 背景重建可以检测到高度异常（地面平面之上的点）。
- **H3**：云端 object-level 3DGS 可以为低矮物体生成更准确的 3D 形状 → BEV footprint 比 LiDAR 点云更完整。

### 指标

| 指标 | 定义 | 预期方向 |
|------|------|----------|
| Obstacle detection rate | 正确识别为障碍物的比例 | Enhanced > Baseline |
| Footprint estimation error | 估计占据区域与实测的 IoU | Enhanced > Baseline |
| Height estimation error | 估计物体最高点与实测的差 (cm) | 仅 Enhanced |
| Unnecessary stop count | (同上) | Enhanced < Baseline |
| Path length | (同上) | Enhanced ≤ Baseline |

### 采集数据

- 同上实验 1 的基础数据
- 额外：RGB-D 深度图（用于验证高度估计）
- 每个低矮障碍物的实测 3D 尺寸（卷尺测量）

### 可视化产出

- 侧视图：展示物体实际高度 vs 估计高度
- BEV 视图：LiDAR-only costmap vs LiDAR + 3DGS costmap，标注漏检和误检
- 深度验证：RGB-D 深度 vs 3R 模型估计深度对比

---

## 实验 3：行人/自行车低速交互

### 场景布置

- **空间**：开放区域或宽通道（模拟快递点附近或教学楼门口）
- **动态元素**：一个行人以慢速（~1 m/s）横穿机器人预定路径
- **变体**：行人替换为慢速自行车（~1.5 m/s）
- **机器人任务**：从 A 点到 B 点，过程中行人/自行车从侧面横穿

### 假设

- **H1**：本地避障（LiDAR + 超声波）可以安全处理动态障碍物（停车等待），但路径恢复可能不自然（反复停顿、路径抖动）。
- **H2**：云端感知在此场景中主要作用不在实时避障（延迟太高），而在交互前后的静态场景理解更准确（交互前预判可通行区域，交互后快速恢复路径）。
- **H3**：云端增强 costmap 对静态障碍物更精确 → 交互后路径恢复更平滑。

### 指标

| 指标 | 定义 | 预期方向 |
|------|------|----------|
| Success rate | (同上) | 两者都应为 100% |
| Recovery path length | 从交互结束到回到原路径的长度 (m) | Enhanced ≤ Baseline |
| Time to resume heading | 从交互结束到恢复原航向的时间 (s) | Enhanced ≤ Baseline |
| Interaction clearance | 与人/自行车的最小距离 (m) | 两者都 > 安全阈值 (1.0m) |
| Path smoothness (post-interaction) | 交互后路径段曲率积分 | Enhanced < Baseline |

### 采集数据

- 同上实验 1 的基础数据
- 行人/自行车的运动轨迹（外部记录：手机 GPS 或 overhead camera）
- 交互开始和结束的时间戳

### 可视化产出

- 时空图：机器人 + 行人的轨迹随时间变化的 BEV 动画
- Costmap 序列：交互前后的局部 costmap 快照
- 路径恢复对比：baseline vs enhanced 在交互后 5 秒内的路径放大图

### 特别注意

这个实验**不是**测试云端能否处理动态障碍物——不能，延迟太高。这个实验测试的是**云端增强的静态场景理解在动态交互前后的价值**。动态避障始终由本地安全回路负责。

---

## 评估工具链

所有实验使用统一的评估脚本（`scripts/eval/`）：

- `eval_path_quality.py` — 路径长度、平滑度、停车次数、时间
- `eval_footprint.py` — footprint IoU、尺寸误差
- `eval_costmap.py` — costmap 差异可视化
- `eval_latency.py` — 云端延迟统计

---

## 实验记录

每个实验运行后，在 `experiments/` 下按模板创建日志：

```
experiments/YYYY-MM-DD_exp1_narrow_passage.md
experiments/YYYY-MM-DD_exp2_low_obstacles.md
experiments/YYYY-MM-DD_exp3_pedestrian_interaction.md
```
