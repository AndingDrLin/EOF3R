# 任务清单

> 按 8 阶段组织。已完成项保留勾选。

---

## Stage 0：项目初始化

- [x] 创建目录结构
- [x] 创建 .gitignore
- [x] 创建 README.md
- [x] 创建 requirements.txt（骨架）
- [x] 创建 docs/project_scope.md
- [x] 创建 docs/roadmap.md
- [x] 创建 docs/lit_review.md
- [x] 创建 docs/standards.md
- [x] 创建 docs/todo.md
- [x] 创建 docs/project_audit.md（方向诊断）
- [x] 创建 docs/experiments.md（导航实验设计）
- [x] 创建 docs/engineering.md（三阶段工程规划）
- [x] 创建 docs/risks.md（风险评估）
- [x] 创建 lit_notes/_template.md
- [x] 创建 experiments/exp_template.md
- [x] 创建 configs/default.yaml
- [x] 更新 configs/default.yaml（加 robot/cloud/costmap/safety）
- [x] 更新 CLAUDE.md（新方向 + 新目录 + BEV 约定）
- [x] 重写 README.md（新方向 + 系统架构）
- [x] 创建 src/__init__.py
- [x] 创建 conda 环境 `eof3r`（Python 3.10 + torch 2.5.1 + CUDA 12.1）
- [x] 测试基础环境可运行（SAM2 + VGGT + torch + numpy + scipy 全部可用）
- [x] Stage 0 收尾 commit

---

## Stage 1：文献调研与基线选定 ✅ 已完成 (2025-06-16)

### 核心论文
- [x] Zhang et al., "Review of Feed-forward 3D Reconstruction" (arXiv 2025) — 通过 Web 调研了解框架
- [x] Kerbl et al., "3DGS" (SIGGRAPH 2023) — 笔记
- [x] Wang et al., "VGGT" (CVPR 2025) — 笔记
- [x] MVSplat (ECCV 2024) — 笔记 + 代码实验（待 Stage 3）
- [x] SAM 2 论文 + 官方文档

### 各方向调研
- [x] A 方向（3DGS 基础）笔记完成：3 篇（3DGS, Mip-Splatting, 2DGS）
- [x] B 方向（Object-level 3DGS）：1 篇笔记（ObjectGS）+ 3 篇待读清单
- [x] C 方向（Feedforward 3R）：3 篇笔记（VGGT, DUSt3R, MASt3R/MUSt3R）
- [x] D 方向（场景分割）调研完成：2 篇笔记（SAM2, YOLO-World）
- [x] E 方向（混合表示）：2 篇笔记（Unbounded-GS, HybridGS）
- [x] F 方向（Autonomous Demo）调研完成：1 篇笔记（GaussianFormer）+ 2 篇待读
- [x] G 方向（Feedforward/Sparse-View 3DGS）：3 篇笔记（MVSplat, pixelSplat, latentSplat）
- [x] H 方向（Semantic 3DGS）：3 篇笔记（LangSplat, LEGaussians, Feature 3DGS）
- [x] I 方向（BEV Occupancy & Costmap）：2 篇笔记（Occ3D, Nav2 costmap_2d docs）
- [x] J 方向（ROS2 Nav2 Local Planning）：1 篇笔记（Nav2 Planning docs）
- [x] K 方向（Edge-Cloud Robotics）：1 篇笔记（FogROS2）
- [x] L 方向（Campus Delivery）：1 篇笔记（Starship）+ Nuro/美团/京东 待深入

### 决策
- [x] 确定各模块 baseline 选择
- [ ] 更新 baselines/registry.yaml（待 Stage 2/3 代码实验后最终确认）
- [x] 更新 lit_review.md 为完整状态

---

## Stage 2：数据准备与场景分解

- [ ] 确认 $EOF3R_DATA 环境变量和数据路径
- [x] 公开数据可用：Re10k 4 帧 720p 样本保存到 `data/public/re10k_samples/`
- [ ] 下载/准备 ScanNet++ 参考数据（至少 2 个场景）
- [ ] 预约 Husky + 校园场地
- [ ] 录制 campus rosbag（至少 3 个场景）
- [ ] 写 rosbag 解析脚本（`eof3r/scripts/preprocess/extract_frames.py`）
- [x] SAM2 分割在 Re10k 上验证（65 object 过分割，需调参或加 YOLO 预处理）
- [ ] 前景 mask 提取 pipeline
- [ ] 物体 crop 提取（多帧关联）
- [ ] 背景 mask 生成
- [ ] Gazebo/Isaac 仿真场景搭建（可选）

---

## Stage 3：前景 Object-level 3DGS 重建

> 状态：SAM2 真模型已验证，MVSplat real 推理待测（env 隔离问题），fusion 瓶颈待解决

- [x] 搭建 MVSplat 推理环境（baselines/mvsplat/ 已 clone + re10k.ckpt）
- [x] 写 MVSplat wrapper（`src/foreground/mvsplat_wrapper.py`）— build/infer/extract_occupancy
- [x] E2E 测试通过（scripts/eval/test_e2e_pipeline.py — 5 阶段定量指标，自动检测真/stub）
- [x] SAM2 clone + SAM2Wrapper（`src/segmentation/sam2_wrapper.py`）— HuggingFace 自动下载
- [x] SAM2 安装依赖（pip install -e baselines/sam2/ → eof3r env, torch 2.5.1）
- [x] SAM2 real 验证：在合成图像上检测到 1 object，6.9s ✅
- [ ] MVSplat real 推理在 eof3r env 中运行（通过 path isolation workaround 可达，但坐标尺度待校准）
- [ ] 测试：单物体 + 2-4 crops → Gaussian .ply（真模型推理 + 坐标校准后）
- [ ] 3D 几何精度评估（Chamfer, F-Score vs GT mesh）
- [x] 物体参数提取：MVSplatWrapper.extract_occupancy() 返回 3D center, size, BEV footprint（需校准坐标）
- [ ] Fallback: per-object 3DGS 优化 wrapper
- [ ] 对比：feedforward vs optimization 的精度/速度

---

## Stage 4：背景 3R 粗重建

> 状态：VGGT-1B 真模型已验证通过 ✅

- [x] VGGT clone + 搭建环境（baselines/vggt/ 已 clone）
- [x] 写 VGGT stub（`src/background/vggt_stub.py`）— 合成点云/位势/地面/可通行
- [x] 写 VGGTWrapper（`src/background/vggt_wrapper.py`）— from_pretrained + 6D 位姿解码 + 地面平面估计
- [x] VGGT 安装依赖（pip install -e baselines/vggt/ → mvsplat env 和 eof3r env 均已装）
- [x] VGGT real 验证：输出 (2, 378, 504, 3) pointmap + 2 帧相机位姿，14.4s ✅
- [ ] 搭建 MASt3R/DUSt3R fallback 环境
- [ ] 坐标系统验证（VGGT 输出帧 → Y-up → Z-up 转换确认）

---

## Stage 5：融合与 BEV 代价地图生成

> 状态：所有模块就绪，真模型 E2E 跑通。当前瓶颈：fusion 矢量化（130s → 目标 <1s）

- [x] 前景-背景坐标对齐验证（Y-up → Z-up 转换 + coord_utils.py）
- [x] Object Gaussian → BEV 占据网格投影实现（src/fusion/bev_projector.py — max/sum/threshold）
- [x] 语义/风险层级生成（src/costmap/costmap_generator.py — semantic_weights dict）
- [x] Costmap inflation 参数调优（Nav2 风格的 maximum_filter inflation）
- [x] Nav2 costmap layer plugin 骨架（src/costmap/ — 输出 uint8 0-254 格式）
- [x] E2E 真模型验证：SAM2 + VGGT-1B → fusion → costmap，pipeline 全链路跑通（但存在坐标质量问题：BEV coverage 0.45%, drivable conflict 100% — 见 docs/current_issues.md）
- [x] **矢量化 fusion BEV 投影** — np.bincount scatter + gaussian_filter，130s → 0.2s（650x）
- [ ] **MVSplat real 推理接入 + 坐标系校准** — 当前 MVSplat 输出坐标尺度与 VGGT 不匹配，需注入 scale/translation 校准
- [ ] 真数据验证（替换合成 FG 为真实 MVSplat 输出 + 坐标系校准后）
- [ ] ROS2 Nav2 节点适配（当前仅生成 numpy costmap，未 publish to ROS topic）

---

## Stage 6：车-云异步架构

- [ ] 通信协议设计（关键帧格式、costmap patch 格式）
- [ ] 车端关键帧选择 + 上传节点
- [ ] 云端推理服务器（HTTP/gRPC）
- [ ] 云端 pipeline 集成（Stage 2-5 串联）
- [ ] 车端 costmap patch 接收 + 融合节点
- [ ] 延迟监测 + 超时丢弃逻辑
- [ ] 通信中断降级 + 重连测试

---

## Stage 7：实验验证与消融

### 实验 1：窄通道 + 不规则障碍物
- [ ] 场景搭建
- [ ] 5 次重复 × (baseline + enhanced) 运行
- [ ] 指标计算 + 可视化
- [ ] 统计检验

### 实验 2：低矮/不规则障碍物
- [ ] 场景搭建
- [ ] 测试运行 + 指标

### 实验 3：行人/自行车低速交互
- [ ] 场景搭建
- [ ] 测试运行 + 指标

### 消融
- [ ] costmap 去掉语义层
- [ ] costmap 去掉物体形状层

---

## Stage 8：论文写作与答辩

- [ ] 论文初稿
- [ ] 定量结果表
- [ ] 可视化图（≥300 DPI）
- [ ] 系统架构图
- [ ] Demo 视频
- [ ] 答辩 PPT
