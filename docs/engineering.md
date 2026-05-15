# 工程实现阶段规划

> 三阶段渐进式实现。每个阶段有独立产出价值，不依赖下一阶段完成。

---

## 总体原则

1. **Phase N 不依赖 Phase N+1**。每一阶段独立可展示、可评估。
2. **本地安全先行**。任何涉及真实机器人的阶段之前，安全机制必须就位。
3. **Cloud is enhancement, not control**。这条原则贯穿所有阶段的设计。
4. **每个模块独立可运行**。不出现"必须启动全部模块才能测试一个模块"的情况。
5. **配置驱动**。所有路径、参数、topic 名称通过 YAML 配置，不硬编码。

---

## Phase 1：离线验证（预计 4-6 周）

### 目标

在离线数据上验证完整 pipeline 的技术可行性：分割 → 前景重建 → 背景重建 → costmap 生成 → 路径规划可视化。

### 不涉及

- ROS2 实时通信
- 车端-云端网络传输
- 真实机器人运动控制
- 在线推理

### 硬件需求

- 一台 GPU 服务器（≥12GB VRAM，用于模型推理）
- 可选：Husky 仅用于数据采集（人工遥控录制 rosbag），不需要自动控制

### 软件组件

| 组件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `scripts/preprocess/extract_frames.py` | 从 rosbag 提取关键帧 | rosbag (.db3) | PNG 图像 + camera_info |
| `src/segmentation/` | SAM2 前景分割 | RGB 图像 | 前景 masks + 物体 crops |
| `src/foreground/` | MVSplat/3DGS 物体重建 | 物体 crops + masks + 位姿 | 物体 .ply + metadata json |
| `src/background/` | VGGT/MASt3R 背景重建 | 全图/背景图 | 背景点云 + 位姿 + 地面 |
| `src/fusion/` | 坐标对齐 + BEV 投影 | 前景/背景 3D 数据 | 统一坐标系下的占据网格 |
| `src/costmap/` | 语义 costmap 生成 | 占据网格 + 语义标签 | Nav2 costmap msg |
| `scripts/eval/` | 离线评估 + 可视化 | costmap + 路径数据 | 指标 json + 对比图 |

### 数据流

```
rosbag → extract_frames
  ├→ segmentation → foreground crops
  │                    └→ foreground/ (3DGS/MVSplat) → object Gaussians
  └→ background/ (VGGT/MASt3R) → pointmap + poses + ground
                                        ↓
              fusion/ (align + BEV projection) → occupancy grid
                                        ↓
              costmap/ (semantic layers + inflation) → Nav2 costmap
                                        ↓
              eval/ (offline path planning + metrics)
```

### 成功标准

- [ ] 完整 pipeline 在至少 3 个场景（可使用 ScanNet++ 室内 + 自采 campus rosbag）上端到端运行
- [ ] 物体级 3DGS 重建在少视角（2-4 crop）下，3D 几何精度（Chamfer/F-Score）可量化评估
- [ ] BEV costmap 输出可在 RViz 中显示并与原始图像对齐验证
- [ ] 离线路径规划（在 costmap 上跑 Nav2 local planner）生成可行路径
- [ ] Baseline vs Enhanced costmap 至少有 1 个场景中产生可见差异

### 转入 Phase 2 的门槛

- Pipeline 端到端延迟 < 30s 每批关键帧（离线可接受，不要求实时）
- 至少 1 个场景的 enhanced costmap 比 baseline costmap 有明显改善
- 物体 BEV footprint 估计误差 < 20cm（在可控场景中）

---

## Phase 2：半在线系统（预计 3-4 周）

### 目标

Husky 实时上传关键帧，云端异步推理并返回 costmap patch，RViz 显示增强 costmap 和路径。车辆仍由人类遥控或本地 Nav2 控制，云端结果只用于可视化和分析。

### 新增内容（相对 Phase 1）

- ROS2 通信框架
- 车端-云端网络传输
- 异步推理 pipeline
- 延迟监测和超时处理

### 硬件需求

- Husky 无人车（带 RGB-D 相机 + LiDAR + IMU + 车载电脑）
- 云端 GPU 服务器（与车端通过 WiFi/4G 通信）
- 人类操作员（遥控或监督）

### 新增软件组件

| 组件 | 功能 | 位置 |
|------|------|------|
| `src/communication/uploader.py` | 关键帧选择 + 异步上传 | 车端 |
| `src/communication/server.py` | 接收请求 → 调用推理 pipeline → 返回结果 | 云端 |
| `src/communication/receiver.py` | 接收 costmap patch + 发布 ROS2 topic | 车端 |
| `src/communication/monitor.py` | 延迟统计 + 超时丢弃 + 心跳 | 车端 |
| `scripts/robot/launch.py` | 启动车端 ROS2 节点 | 车端 |
| `configs/cloud/server.yaml` | 云端服务配置 | 云端 |
| `configs/robot/husky.yaml` | 车端传感器和通信配置 | 车端 |

### 数据流

```
车端 Husky                             云端 Server
──────────                             ──────────
相机图像 ─→ 关键帧选择 ─→ HTTP/gRPC ─→ 推理 pipeline
                                      (seg/foreground/background/fusion/costmap)
                                        ↓
本地 costmap ←─ ROS2 topic ←─ HTTP/gRPC ←─ costmap patch (json/protobuf)
    +
  RViz 显示
```

### 通信协议

- **上传**：JPEG 图像（质量 80）+ 时间戳 + 相机内参 + 可选 odometry。JSON body + base64 图像。
- **下载**：costmap grid (numpy → base64) + metadata（分辨率、原点、时间戳、语义层）。
- **超时**：车端等待 3s，超时丢弃。不重试。
- **心跳**：车端每秒检查连接，连续 5s 无响应视为断连，降级为本地模式。

### 成功标准

- [ ] 云端 costmap patch 在 RViz 中显示，延迟测量有记录
- [ ] 最佳情况下（本地 WiFi 热点）端到端延迟 < 3s
- [ ] 超时丢弃逻辑正确（模拟网络延迟或云端慢响应）
- [ ] 通信中断后车端不崩溃，降级为本地模式，恢复后自动重连
- [ ] 系统连续运行 10 分钟以上不崩溃、不内存泄漏

### 转入 Phase 3 的门槛

- 端到端延迟 p50 < 3s（在校园 WiFi 环境下）
- 系统稳定运行 > 10 分钟
- 安全机制全部就位并测试通过（急停、速度限制、心跳）
- 获得导师/实验室批准进行 Phase 3 测试

---

## Phase 3：低速闭环 Demo（预计 2-3 周，STRETCH GOAL）

### 目标

Husky 使用 Nav2 + 云端增强 costmap 自主导航。速度限制 0.2-0.5 m/s。人类操作员始终在场并手持急停。

### 新增内容（相对 Phase 2）

- Nav2 完整配置（global planner + local planner + cloud costmap layer）
- 实际车辆运动控制
- 安全监控节点
- 紧急停止机制

### 安全措施（必须全部就位）

| 措施 | 实现 | 触发条件 |
|------|------|----------|
| 物理急停 | 硬件急停开关 | 人类操作员手动触发 |
| 软件急停 | ROS2 safety controller | 心跳丢失、速度超限、碰撞检测 |
| 速度硬限制 | safety controller | cmd_vel 中线速度/角速度超过阈值 |
| 云端超时降级 | communication/monitor.py | 云端心跳丢失 > 5s |
| 电池监控 | Husky 自带 | 低电量自动停车 |
| bumper 停车 | Husky 自带 | 物理碰撞立即停车 |

### 新增软件组件

| 组件 | 功能 |
|------|------|
| `src/demo/safety_controller.py` | 速度限制 + 心跳监控 + 急停 relay |
| `src/demo/costmap_fusion.py` | 融合本地 + 云端 costmap layer |
| `scripts/robot/nav2_params.yaml` | Nav2 完整参数（含 cloud layer） |
| `scripts/robot/husky_launch.py` | 完整系统 launch |

### 测试流程

1. **空场地测试**：无障碍物，验证基础导航（全局规划 + 局部规划）正常
2. **静态障碍物测试**：简单障碍物（纸箱），验证云端增强 costmap 被 Nav2 使用
3. **实验场景测试**：按 `docs/experiments.md` 中的 3 个场景逐一测试
4. **降级测试**：手动断开云端通信，验证车辆降速/停车/恢复

### 成功标准

- [ ] 至少 1 个实验场景完成完整闭环测试
- [ ] 无安全事件
- [ ] Baseline vs Enhanced 路径质量差异可量化
- [ ] 降级测试通过

### 如果 Phase 3 不做

Phase 1 + 2 的结果 + 详细的 Phase 3 设计 + 降级行为分析 → 作为论文主体。Phase 3 作为 "future work" 讨论。

---

## 关键接口定义

所有阶段共享的接口，在 Phase 1 就确定：

```
# 物体 3DGS 输出格式 (json)
{
  "object_id": int,
  "class": str,            # "bicycle", "cone", "box", ...
  "risk_level": int,       # 0-3, 0=可通行, 3=必须远离
  "center_3d": [x, y, z],  # meters, world frame
  "size_3d": [l, w, h],    # meters
  "orientation": [qw, qx, qy, qz],
  "bev_footprint": [[x1,y1], [x2,y2], ...],  # polygon vertices
  "confidence": float,
  "ply_path": str           # relative path to .ply file
}

# Costmap patch 格式 (ros2 msg 或 json)
{
  "timestamp": float,
  "resolution": float,      # m/cell
  "width": int, "height": int,
  "origin": [x, y, theta],
  "data": [int8 array],     # 0-100 occupancy probability
  "semantic_layer": {       # optional
    "class_ids": [int array],
    "risk_levels": [int array]
  }
}
```

---

## 代码模块与阶段的对应

| 模块 | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| `src/segmentation/` | ✓ | 复用 | 复用 |
| `src/foreground/` | ✓ | 复用 | 复用 |
| `src/background/` | ✓ | 复用 | 复用 |
| `src/fusion/` | ✓ | 复用 | 复用 |
| `src/costmap/` | ✓ | 复用 | 复用 |
| `src/communication/` | — | ✓ | 复用 |
| `src/demo/` | — | — | ✓ |
| `src/utils/` | ✓ | ✓ | ✓ |
| `scripts/preprocess/` | ✓ | 复用 | 复用 |
| `scripts/eval/` | ✓ | ✓ | ✓ |
| `scripts/robot/` | — | ✓ | ✓ |
