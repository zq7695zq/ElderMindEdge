# 动作识别评估系统

该评估系统为动作识别系统在不同推理模式和数据集上提供客观评估。

## 功能特性

- **多数据集支持**: NTU RGB+D 和 AIGC 数据集
- **三种推理模式**: 本地推理、云推理和混合推理
- **客观指标**: 延迟和准确率测量
- **全面报告**: JSON 和文本格式的结果输出

## 使用方法

### 命令行界面

```bash
# 在 NTU RGB+D 数据集上使用本地推理进行评估
python main.py --mode evaluate --dataset-path datasets/nturgb+d_rgb --dataset-type ntu-rgbd --inference-mode local

# 在 AIGC 数据集上使用云推理进行评估
python main.py --mode evaluate --dataset-path datasets/aigc --dataset-type aigc --inference-mode cloud

# 使用混合推理进行评估（默认）
python main.py --mode evaluate --dataset-path datasets/nturgb+d_rgb --dataset-type ntu-rgbd --inference-mode hybrid
```

### 数据集结构

#### NTU RGB+D 数据集
```
datasets/nturgb+d_rgb/
├── S017C003P020R002A059_rgb.avi
├── S017C003P020R002A001_rgb.avi
└── ...
```

文件名格式: `S{setup}C{camera}P{performer}R{replication}A{action}_rgb.avi`
- 动作ID从 `A{action}` 部分提取

#### AIGC 数据集
```
datasets/aigc/
├── A7_throw/
│   ├── 01/
│   │   ├── video.mp4
│   │   └── metadata.txt
│   ├── 02/
│   │   ├── video.mp4
│   │   └── metadata.txt
│   └── ...
└── A47_touch_neck/
    └── ...
```

每个样本目录包含:
- 一个视频文件 (MP4/AVI/MOV)
- 一个包含动作标签信息的 `metadata.txt` 文件

## 推理模式

### 本地推理
- 仅使用本地 SkateFormer 模型
- 无 LLM 或云处理
- 最快但对复杂动作的准确率可能较低
- 为客观评估禁用所有优化功能

### 云推理
- 将整个视频上传到云 LLM 进行分析
- 无本地动作识别处理
- 延迟最高但对复杂场景的准确率可能更好
- 需要 LLM API 配置

### 混合推理
- 使用现有系统，同时进行本地和云处理
- 本地 SkateFormer 提供初始检测
- 云 LLM 为不确定情况提供额外分析
- 平衡速度和准确率

## 评估指标

### 延迟指标
- **平均延迟**: 从视频开始到结果的平均时间
- **中位延迟**: 所有样本的中位延迟
- **最小/最大延迟**: 延迟值范围
- **标准差**: 延迟的变异性

### 准确率指标
- **总体准确率**: 正确预测的百分比
- **每类准确率**: 每个动作类别的准确率
- **精确率/召回率/F1**: 标准分类指标
- **混淆矩阵**: 详细的预测分析

## 输出文件

评估系统生成三种类型的输出文件:

1. **detailed_results.json**: 每个样本的完整结果
2. **metrics_summary.json**: JSON 格式的聚合指标
3. **evaluation_report.txt**: 人类可读的摘要报告

## 配置

评估系统使用与主系统相同的配置文件 (`configs/stream_config.yaml`)，但会自动禁用某些功能以进行客观评估:

- 禁用视频录制
- 禁用推理调度器优化（本地模式）
- 禁用目标动作增强（本地模式）
- 禁用 LLM 推理（仅本地模式）

## 示例输出

```
评估摘要
============================================================
推理模式: local
数据集: ntu-rgbd
总样本数: 100
评估时长: 45.23 秒

延迟指标:
------------------------------
平均延迟: 0.452 秒
中位延迟: 0.398 秒
最小延迟: 0.234 秒
最大延迟: 1.234 秒
标准差: 0.156 秒

准确率指标:
------------------------------
总体准确率: 0.850 (85.0%)
正确预测: 85/100
```

## 系统要求

- 主动作识别系统的所有依赖项
- 正确格式的数据集文件
- 对于云推理: 有效的 LLM API 配置
- 足够的磁盘空间用于结果文件

## 注意事项

- 评估系统设计为客观的，不使用任何可能偏向结果的优化技术
- 每种推理模式都是独立的，以提供公平比较
- 结果会自动添加时间戳以避免冲突
- 大型数据集可能需要大量时间处理，特别是云推理
