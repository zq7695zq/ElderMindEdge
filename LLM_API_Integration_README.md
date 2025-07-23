# LLM API 集成说明

## 概述

已成功将原有的复杂LLM推理系统替换为调用您指定的 `/api/llm/inference` 接口的简化实现。新系统会自动将视频上传到OSS，然后将OSS URL发送给API接口进行推理。

## 系统架构

```
事件触发 -> 视频上传到OSS -> 调用/api/llm/inference -> 返回推理结果
```

## 主要组件

### 1. LLMAPIClient (`utils/llm_api_client.py`)
- 负责调用 `/api/llm/inference` 接口
- 集成视频上传功能
- 支持重试机制和错误处理
- 根据配置判断是否触发推理

### 2. VideoUploader (`utils/video_uploader.py`)
- 负责将本地视频文件上传到阿里云OSS
- 返回OSS URL供API接口使用
- 支持文件管理和清理

## API 接口调用格式

### 请求
```http
POST /api/llm/inference
Content-Type: application/json

{
  "videoUrl": "https://my-llm-server.oss-cn-guangzhou.aliyuncs.com/llm_videos/1642781234_event_video.mp4",
  "customPrompt": "请分析这个视频中的动作行为。检测到的动作：falling，置信度：0.65"
}
```

**注意**: 请求格式已更新为匹配Spring Boot接口的驼峰命名规范：
- `video_url` → `videoUrl`
- `prompt` → `customPrompt`
- 移除了 `event_info` 对象，事件信息现在包含在 `customPrompt` 中

### 响应
期望返回JSON格式的推理结果。

## 配置说明

在 `configs/stream_config.yaml` 中的配置：

```yaml
llm_inference:
  enabled: true  # 是否启用LLM推理
  api_url: "http://localhost:8080/api/llm/inference"  # API接口地址
  timeout: 60  # 请求超时时间(秒)
  max_retries: 3  # 最大重试次数
  retry_delay: 1.0  # 重试延迟时间(秒)
    
  # 推理触发条件
  trigger_conditions:
    all_events: false  # 是否对所有事件进行推理
    target_actions_only: false  # 仅对目标动作进行推理
    min_confidence: 0.0  # 最小置信度阈值
    max_confidence: 0.8  # 最大置信度阈值
    critical_actions_only: false  # 仅对关键动作进行推理
  
  # 视频上传配置
  video_upload:
    enabled: true  # 是否启用视频上传
    cloud_provider: "oss"  # 云存储提供商
    oss_config:
      access_key_id: "YOUR_ACCESS_KEY_ID"
      access_key_secret: "YOUR_ACCESS_KEY_SECRET"
      endpoint: "https://oss-cn-guangzhou.aliyuncs.com"
      bucket_name: "my-llm-server"
```

## 工作流程

1. **事件触发**: 当动作识别系统检测到事件时
2. **触发条件判断**: 根据配置判断是否需要LLM推理
3. **视频上传**: 将事件视频片段上传到OSS
4. **API调用**: 使用OSS URL调用 `/api/llm/inference` 接口
5. **结果处理**: 处理API返回的推理结果
6. **回调通知**: 通过回调函数通知上层应用

## 触发条件

系统会根据以下条件判断是否触发LLM推理：

- **视频文件存在**: 必须有有效的视频文件路径
- **置信度区间**: 在 `min_confidence` 和 `max_confidence` 之间
- **特定动作**: 可配置仅对目标动作或关键动作进行推理
- **全局开关**: 可配置对所有事件进行推理

## 依赖库

需要安装以下Python库：
```bash
pip install oss2  # 阿里云OSS SDK
pip install requests  # HTTP请求库
```

## 测试

运行测试脚本验证功能：
```bash
python test_llm_api.py
```

## 注意事项

1. **OSS配置**: 确保OSS配置正确，包括访问密钥、端点和存储桶名称
2. **网络连接**: 确保能够访问OSS和您的API服务器
3. **视频格式**: 确保上传的视频格式被您的API接口支持
4. **存储成本**: 注意OSS存储和流量成本
5. **清理机制**: 可以考虑定期清理OSS中的旧视频文件

## 与原系统的主要差异

| 方面 | 原系统 | 新系统 |
|------|--------|--------|
| 复杂度 | 高（多种推理模式、限流器等） | 低（简单HTTP调用） |
| 依赖 | 智谱AI SDK、本地模型等 | 仅需requests和oss2 |
| 配置 | 复杂的多层配置 | 简化的配置 |
| 维护性 | 需要维护多个组件 | 仅需维护API调用逻辑 |
| 扩展性 | 受限于本地资源 | 依赖于API服务的扩展性 |
