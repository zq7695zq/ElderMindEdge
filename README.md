# 🎯 YOLO + SkateFormer Action Recognition System

Real-time action recognition system combining YOLO pose estimation with SkateFormer transformer architecture for video analysis.

## ✨ Features

- **Real-time Processing**: Multi-threaded pipeline for video files, webcams, and RTSP streams
- **Action Enhancement**: Priority-based boosting for target actions
- **Event Recording**: Automatic video clip generation with pre/post-event buffers
- **LLM Integration**: Support for ZhipuAI GLM-4V and local models
- **Cloud Storage**: Alibaba Cloud OSS, Tencent COS, and AWS S3 support

## � Installation

```bash
# Clone repository
git clone <repository-url>
cd Yolo+SkateFormer

# Install dependencies
pip install -r requirements.txt

# Optional: LLM and cloud storage
pip install zhipuai>=2.0.0 oss2>=2.15.0 boto3>=1.26.0
```

**Requirements**: Python 3.8+, FFmpeg, CUDA (optional)

## 🚀 Quick Start

### Python API

```python
from action_recognition_stream import ActionRecognitionStream, ActionEvent

def on_action_event(event: ActionEvent):
    print(f"🎯 {event.action_name} ({event.confidence:.2f}) at {event.timestamp:.1f}s")

stream = ActionRecognitionStream(
    config_path="configs/stream_config.yaml",
    event_callback=on_action_event
)

stream.start_stream("test107voice.mp4")  # Video file
# stream.start_stream(0)                  # Webcam
# stream.start_stream("rtsp://...")       # RTSP stream
```

### Command Line

```bash
# Basic usage
python main.py --input-video test107voice.mp4 --config configs/stream_config.yaml

# With custom settings
python main.py --input-video test107voice.mp4 \
    --config configs/stream_config.yaml \
    --boost-factor 4.0 \
    --target-actions 42 8 0 7
```

## 🧪 Testing

```bash
# Test with video file
python main.py --input-video test107voice.mp4 --config configs/stream_config.yaml

# Test with webcam
python main.py --input-video 0 --config configs/stream_config.yaml
```

## ⚙️ Configuration

Main configuration file: `configs/stream_config.yaml`

Key settings:
- **Video processing**: FPS target, confidence thresholds
- **Action enhancement**: Target actions with priority-based boosting
- **Event recording**: Pre/post-event buffers, output directory
- **LLM integration**: API keys, rate limiting, trigger conditions
- **Performance**: GPU settings, adaptive frame skipping

## 📚 API Reference

### ActionRecognitionStream

Main class for video processing:

```python
# Initialize
stream = ActionRecognitionStream(
    config_path="configs/stream_config.yaml",
    event_callback=callback_function
)

# Control
stream.start_stream(source)  # Video file, webcam (0), or RTSP URL
stream.stop_stream()
stream.get_status()
```

### ActionEvent

Event data structure:
- `timestamp`: Event time in seconds
- `action_id`: Action class ID (0-59 or 0-119)
- `action_name`: Human-readable action name
- `confidence`: Prediction confidence (0.0-1.0)
- `enhanced`: Whether action was boosted

## 🚀 Architecture

Multi-threaded pipeline: Video Input → Frame Capture → Skeleton Detection → Inference → Event Processing

**Optimizations**:
- GPU acceleration with CUDA
- Adaptive frame skipping
- Zero-copy frame passing
- Smart memory management

## 🔧 Troubleshooting

Common issues:
- **CUDA out of memory**: Reduce batch size or use CPU
- **FFmpeg not found**: Install FFmpeg and add to PATH
- **Model files missing**: Check paths and download required models

## 📄 License

Uses models from:
- [YOLO](https://github.com/ultralytics/ultralytics) - AGPL-3.0
- [SkateFormer](https://github.com/SAKOFEDRA/SkateFormer) - Check original license
