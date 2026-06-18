# FogROS2: An Adaptive Platform for Cloud and Fog Robotics Using ROS 2

- **作者**：Kaiyuan Chen, Jeffrey Ichnowski, John Kubiatowicz, Ken Goldberg et al. (UC Berkeley)
- **会议/期刊**：IEEE ICRA 2023
- **年份**：2023
- **链接**：[arXiv:2205.09778](https://arxiv.org/abs/2205.09778)
- **代码**：[github.com/BerkeleyAutomation/FogROS2](https://github.com/BerkeleyAutomation/FogROS2) (ROS2 apt: ros-humble-fogros2)
- **阅读日期**：2025-06-16

## 一句话总结

将 ROS2 节点透明地卸载到云端：通过扩展的 ROS2 launch 系统指定哪些节点跑云端、Kubernetes 编排、WireGuard VPN 安全通信、H.264 自适应的图像压缩，实现 SLAM 延迟降低 50%、抓取规划从 14s 加速到 1.2s、运动规划加速 28-45x。

## 核心方法

1. **透明节点卸载**：在 launch 文件中通过特殊语法标记应跑在云端的节点，FogROS2 自动处理网络、容器部署、DDS 发现。
2. **Kubernetes Backend**：替代 FogROS1 的 AWS-only 方案，支持多云（AWS/GCP/Azure）。
3. **WireGuard VPN over UDP**：安全的跨网络 DDS 通信（ROS2 的 topic/service/action 在云和机器人之间透明路由）。
4. **H.264 视频压缩**：自定义 image_transport_plugin 自动压缩/解压图像 topic，减少 97% 图像往返延迟。
5. **自动化云资源选择**：自动检测最近 AWS region、选择便宜/最优实例类型、自动构建 VM 镜像。

## 关键数字

| 指标 | 值 |
|------|-----|
| SLAM 延迟降低 | 50% |
| Dex-Net 抓取规划 | 14s → 1.2s (11.7x) |
| 运动规划加速 | 28-45x |
| 图像往返延迟降低 | 97% (H.264) |
| 启动时间改善 | 63% |
| 网络利用率降低 | 3.8x |

## 与本文的关系

**架构参考，非直接使用**。FogROS2 提供了车-云通信的 infrastructure 参考（VPN、压缩、容器化、多云编排），但本项目不直接使用 FogROS2（太重）。我们借鉴其设计模式：H.264 压缩图像上传、VPN 安全通信、云端容器化推理服务。本项目的定制通信模块比 FogROS2 更轻量、更专注：我们只需要上传关键帧+下发 costmap patch，不需要完整的 ROS2 节点透明卸载。

## 可用性

- [x] 代码开源
- [x] ROS2 apt 可安装
- [ ] 已在本地跑通
- [x] N/A（非 ML 模型）

## 笔记

- FogROS2 最重要的教训：**图像压缩是关键**——原始图像 topic 的网络带宽消耗巨大，H.264 压缩将延迟降低 97%。本项目的图像上传必须做压缩。
- FogROS2 的延迟数字对设定本项目预期有帮助：SLAM（计算密集型）延迟降低 50%，但绝对延迟仍然 >100ms。这印证了我们的设计——云端用于异步增强而非实时控制。
- FogROS2-Sky (ICRA 2024) 的"自动选择最便宜-最低延迟云端配置"能力，在生产环境中可能有用（多机器人部署时优化成本）。
