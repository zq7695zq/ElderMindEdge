# Action Recognition Stream

An optimized real-time action recognition system that processes video streams (RTSP, webcam, or video files) and generates action events using YOLO pose estimation and SkateFormer action recognition.

## Features

- **Real-time processing**: Optimized for low-latency streaming
- **RTSP support**: Can process RTSP video streams
- **Event-driven**: Converts video streams into action events
- **Target action enhancement**: Boost specific actions for better detection
- **Multi-source support**: Works with video files, webcams, and RTSP streams
- **Efficient preprocessing**: Optimized skeleton data processing
- **Thread-safe**: Uses threading for smooth real-time performance

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg (for RTSP testing)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the required model files:
   - `yolo11x-pose.pt` (YOLO pose estimation model)
   - `configs/SkateFormer_j.yaml` (SkateFormer configuration)
   - `pretrained/ntu60_CView/SkateFormer_j.pt` (SkateFormer weights)

## Quick Start

### Basic Usage with Configuration File (Recommended)

```python
from action_recognition_stream import ActionRecognitionStream, ActionEvent

def on_action_event(event: ActionEvent):
    enhanced_str = " [ENHANCED]" if event.enhanced else ""
    print(f"Action: {event.action_name} ({event.confidence:.2f}){enhanced_str}")

# Initialize stream with configuration file
stream = ActionRecognitionStream(
    config_path="configs/enhanced_stream_config.yaml",
    event_callback=on_action_event
)

# Start processing
stream.start_stream("your_video.mp4")  # or webcam: 0, or RTSP: "rtsp://..."

# Stop when done
stream.stop_stream()
```

### Advanced Usage with Parameter Overrides

```python
# Override specific settings while using config file
stream = ActionRecognitionStream(
    config_path="configs/enhanced_stream_config.yaml",
    target_actions=[0, 42, 9],  # Override target actions
    boost_factor=4.0,           # Override boost factor
    fps_target=25,              # Override FPS target
    event_callback=on_action_event
)
```

### Legacy Usage (Direct Parameters)

```python
# Direct parameter specification (without config file)
stream = ActionRecognitionStream(
    yolo_model_path="yolo11x-pose.pt",
    skateformer_config_path="configs/SkateFormer_j.yaml",
    skateformer_weights_path="pretrained/ntu60_CView/SkateFormer_j.pt",
    target_actions=[0, 42],  # drink water, falling
    boost_factor=3.0,
    event_callback=on_action_event
)
```

### Run Example

```bash
# Use configuration file
python example_usage.py

# Use command line with config file
python main.py --input-video test107voice.mp4 --config configs/enhanced_stream_config.yaml

# Override specific settings
python main.py --input-video test107voice.mp4 --config configs/enhanced_stream_config.yaml --boost-factor 4.0 --target-actions 0 42 9
```

## Testing

### Test with Video File
```bash
python test.py
```

This will run two tests:
1. Direct video file processing
2. RTSP stream processing (requires FFmpeg)

### Test Requirements

- FFmpeg installed and in PATH (for RTSP testing)
- Test video file: `test107voice.mp4`

## Configuration

### Configuration File

The system uses YAML configuration files to manage settings. The main configuration file is `configs/enhanced_stream_config.yaml`:

```yaml
stream_config:
  # Action enhancement settings
  target_actions:
    enabled: true
    boost_factor: 3.0
    actions:
      - id: 42
        name: "falling"
        priority: critical  # 5.0x boost
      - id: 0
        name: "drink water"
        priority: high      # 3.0x boost
      - id: 4
        name: "drop"
        priority: medium    # 2.0x boost
      - id: 9
        name: "clapping"
        priority: low       # 1.5x boost
  
  # Priority boost factors
  priority_boost_factors:
    critical: 5.0
    high: 3.0
    medium: 2.0
    low: 1.5
  
  # Event filtering
  event_filtering:
    enabled: true
    min_confidence: 0.15
    duplicate_suppression: true
    duplicate_time_window: 1.0
```

### Target Actions with Priorities

The enhanced system supports priority-based action enhancement:

- **Critical Priority (5.0x boost)**: Safety-related actions like falling, staggering
- **High Priority (3.0x boost)**: Important actions like drinking water, aggressive behavior
- **Medium Priority (2.0x boost)**: Moderate interest actions like pickup, drop, phone calls
- **Low Priority (1.5x boost)**: General interest actions like clapping, waving

### Event Categories

Actions are grouped into categories for better organization:

```yaml
event_categories:
  safety:
    - 42  # falling
    - 41  # staggering
    - 49  # punching/slapping
  interaction:
    - 57  # handshaking
    - 54  # hugging
    - 22  # hand waving
  daily_activities:
    - 0   # drink water
    - 4   # drop
    - 5   # pickup
```

### Configuration Files

- `configs/stream_config.yaml`: Basic configuration
- `configs/enhanced_stream_config.yaml`: Enhanced configuration with detailed action priorities
- `configs/SkateFormer_j.yaml`: SkateFormer model configuration

### Event Filtering

Enhanced event filtering system:

```yaml
event_filtering:
  enabled: true
  min_confidence: 0.15          # Minimum confidence threshold
  duplicate_suppression: true   # Avoid duplicate events
  duplicate_time_window: 1.0    # Time window for duplicate detection
  max_events_per_second: 10     # Rate limiting
```

## API Reference

### ActionRecognitionStream

Main class for processing video streams.

#### Constructor
```python
ActionRecognitionStream(
    yolo_model_path: str,
    skateformer_config_path: str,
    skateformer_weights_path: str,
    target_actions: List[int] = [0, 42],
    boost_factor: float = 3.0,
    fps_target: int = 30,
    event_callback: Optional[Callable[[ActionEvent], None]] = None
)
```

#### Methods
- `start_stream(stream_source: str) -> bool`: Start processing
- `stop_stream()`: Stop processing
- `get_status() -> StreamStatus`: Get current status
- `get_stats() -> Dict`: Get statistics

### ActionEvent

Event data structure for action recognition results.

#### Properties
- `timestamp: float`: Event timestamp
- `action_id: int`: Action class ID
- `action_name: str`: Human-readable action name
- `confidence: float`: Prediction confidence (0-1)
- `enhanced: bool`: Whether this was a target action
- `frame_id: int`: Frame number
- `bbox: Optional[Tuple]`: Bounding box (future use)

## Performance Optimization

### GPU Usage
- Automatically uses CUDA if available
- Models are loaded on GPU for faster inference

### Memory Management
- Efficient buffering with fixed-size windows
- Pre-allocated arrays to reduce garbage collection
- Optimized skeleton data processing

### Real-time Performance
- Threading for non-blocking processing
- Frame rate control to match target FPS
- Optimized inference pipeline

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **FFmpeg not found**: Install FFmpeg and add to PATH
3. **Model files not found**: Check paths and download required models
4. **RTSP connection failed**: Verify RTSP URL and network connectivity

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Supported Actions

The system supports NTU RGB+D dataset actions:
- NTU60: 60 action classes
- NTU120: 120 action classes

Common actions include:
- drink water, eat meal, brushing teeth
- clapping, hand waving, pointing
- sitting down, standing up, falling
- pickup, drop, throw
- and many more...

## File Structure

```
Yolo+SkateFormer/
├── action_recognition_stream.py  # Main stream processing class
├── example_usage.py             # Simple usage example
├── test.py                      # Comprehensive testing
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── configs/                    # Model configurations
├── pretrained/                 # Pre-trained model weights
└── model/                      # Model definitions
```

## License

This project uses models and code from:
- [YOLO](https://github.com/ultralytics/ultralytics) - AGPL-3.0
- [SkateFormer](https://github.com/SAKOFEDRA/SkateFormer) - Check original license

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
