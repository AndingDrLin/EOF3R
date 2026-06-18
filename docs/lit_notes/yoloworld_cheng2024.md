# YOLO-World: Real-Time Open-Vocabulary Object Detection

- **作者**：Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan
- **会议/期刊**：CVPR 2024
- **年份**：2024
- **链接**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html)
- **代码**：[github.com/AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- **阅读日期**：2025-06-16

## 一句话总结

将 YOLO 系列的实时检测能力与 CLIP 视觉-语言模型结合，通过 Prompt-then-Detect 范式和可重参数化的视觉-语言路径聚合网络（RepVL-PAN），实现 52 FPS 的开放词汇目标检测。

## 核心方法

1. **Prompt-then-Detect 范式**：推理前将用户指定的类别名通过 CLIP 文本编码器离线编码成语义 embedding，推理时将文本 embedding "烘焙"进模型权重（re-parameterization），消除在线文本编码的开销。
2. **RepVL-PAN**：在 YOLO 的 PAN（Path Aggregation Network）中融合多尺度视觉特征和文本 embedding，实现 vision-language 特征融合。
3. **Region-Text Contrastive Loss**：对比学习对齐 region proposal 特征和对应的文本描述。

## 关键数字

| 指标 | 值 |
|------|-----|
| LVIS 零样本 AP (Large) | 35.4 |
| 推理速度 (V100) | 52.0 FPS |
| vs GLIP 速度 | ~400x faster |
| 参数量 (Large) | ~100M |

## 与本文的关系

YOLO-World 是本项目前景分割 pipeline 中 **提供语义标签的推荐候选**。与 SAM2 形成互补：YOLO-World 快速检测物体并提供 box + 类别标签（"知道这是什么"），SAM2 基于 box prompt 生成精细 mask（"知道这在哪"）。Phase 1 的预定义类别集合完全在 YOLO-World 的能力范围内。

## 可用性

- [x] 代码开源
- [x] 权重可下载
- [ ] 已在本地跑通
- [x] 显存要求可接受

## 笔记

- YOLO-World 和 SAM2 是**互补关系**而非替代关系：YOLO-World 输出粗略的检测框 + 语义类别，SAM2 输出精细的像素级 mask 但不知道类别。本项目 Phase 1 的推荐 pipeline：YOLO-World → box proposals（带类别）→ SAM2 box-prompted mask refinement → per-object masks + labels。
- Prompt-then-Detect 范式对本项目非常友好——预定义类别（行人、自行车、路锥、纸箱等）在部署前就已知，可以预编码为 embedding，推理速度极快。
- 如果 YOLO-World 的 AP 在本项目目标场景上不够高，YOLOv8/v11 的固定类别检测是可靠的 fallback。
