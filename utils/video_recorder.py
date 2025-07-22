import cv2
import numpy as np
import json
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# --- 假设这些数据类已经定义在别处 ---
@dataclass
class ActionEvent:
    """Action recognition event data structure"""
    timestamp: float
    action_id: int
    action_name: str
    confidence: float
    enhanced: bool = False
    frame_id: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    video_path: Optional[str] = None  # Path to saved video clip

@dataclass
class FrameData:
    """帧数据结构"""
    frame: np.ndarray
    frame_id: int
    timestamp: float
# -----------------------------------------

logger = logging.getLogger(__name__)

class FrameBasedVideoRecorder:
    """
    基于帧计数的视频录制器，确保准确的事件前后时间。
    增强了对视频流结束的处理能力。
    """

    def __init__(self,
                 output_dir: str = "event_clips",
                 pre_event_frames: int = 150,  # 5秒 × 30fps = 150帧
                 post_event_frames: int = 150,  # 5秒 × 30fps = 150帧
                 fps: int = 30,
                 max_buffer_size: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.pre_event_frames = pre_event_frames
        self.post_event_frames = post_event_frames
        self.fps = fps

        # 使用帧ID而非时间戳的环形缓冲区
        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.frame_ids = deque(maxlen=max_buffer_size)

        # 活跃录制跟踪
        self.active_recordings: Dict[float, Dict] = {}  # {event.timestamp: recording_info}
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        logger.info(f"Frame-based video recorder initialized: {output_dir}")
        logger.info(f"Pre-event: {pre_event_frames} frames ({pre_event_frames/fps:.1f}s), "
                    f"Post-event: {post_event_frames} frames ({post_event_frames/fps:.1f}s)")
        logger.info(f"Frame buffer size: {max_buffer_size} frames ({max_buffer_size/fps:.1f}s)")


    def add_frame(self, frame: np.ndarray, frame_id: int):
        """添加帧到缓冲区，使用帧ID而非时间戳"""
        self.frame_buffer.append(frame.copy())
        self.frame_ids.append(frame_id)

    def start_recording(self, event: ActionEvent) -> Optional[str]:
        """开始录制，基于帧ID而非时间戳"""
        timestamp_str = datetime.fromtimestamp(event.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_action_{event.action_id}_{event.action_name.replace(' ', '_')}.mp4"
        output_path = self.output_dir / filename

        # 计算pre-event帧的起始ID
        event_frame_id = event.frame_id
        pre_event_start_id = max(0, event_frame_id - self.pre_event_frames)

        # 从缓冲区中找到pre-event帧
        pre_event_frames_to_write = []
        # 从后向前遍历缓冲区以提高效率
        for i in range(len(self.frame_ids) - 1, -1, -1):
            if self.frame_ids[i] < pre_event_start_id:
                # 我们已经找到了所有需要的帧
                pre_event_frames_to_write = list(self.frame_buffer)[i+1:]
                break
        else:
            # 如果循环正常结束（未break），说明所有缓冲区的帧都在窗口内
            pre_event_frames_to_write = list(self.frame_buffer)
        
        if not pre_event_frames_to_write:
            logger.warning(f"无法为事件 {event.action_name} 启动录制，因为在帧ID {event.frame_id} 时帧缓冲区为空。")
            return None

        height, width = pre_event_frames_to_write[0].shape[:2]
        try:
            video_writer = cv2.VideoWriter(str(output_path), self.fourcc, self.fps, (width, height))
            if not video_writer.isOpened():
                raise IOError(f"cv2.VideoWriter 无法打开文件: {output_path}")
        except (IOError, cv2.error) as e:
            logger.error(f"创建视频写入器失败: {e}")
            return None

        # 写入pre-event帧
        for frame in pre_event_frames_to_write:
            video_writer.write(frame)

        # 创建录制信息
        self.active_recordings[event.timestamp] = {
            'video_writer': video_writer,
            'output_path': str(output_path),
            'event': event,
            'frames_written': len(pre_event_frames_to_write),
            'end_frame_id': event_frame_id + self.post_event_frames,
            'start_frame_id': self.frame_ids[len(self.frame_ids) - len(pre_event_frames_to_write)] if pre_event_frames_to_write else event_frame_id,
        }

        logger.info(f"Started recording for event: {event.action_name} -> {filename}")
        logger.info(f"Recording frames from ID ~{self.active_recordings[event.timestamp]['start_frame_id']} "
                    f"to {event_frame_id + self.post_event_frames}")
        return str(output_path)

    def update_recordings(self, frame: np.ndarray, frame_id: int) -> List[Dict]:
        """
        用一个新帧更新所有活跃的录制。
        如果录制完成，则将其关闭。
        """
        completed_recordings_info = []
        events_to_remove = []

        for event_time, recording_info in self.active_recordings.items():
            if frame_id <= recording_info['end_frame_id']:
                recording_info['video_writer'].write(frame)
                recording_info['frames_written'] += 1
            else:
                # 录制完成，最终化它
                final_info = self._finalize_recording(recording_info)
                completed_recordings_info.append(final_info)
                events_to_remove.append(event_time)

        for event_time in events_to_remove:
            del self.active_recordings[event_time]

        return completed_recordings_info

    # 新实现的方法
    def has_active_recordings(self) -> bool:
        """检查是否有任何活跃的录制任务。"""
        return bool(self.active_recordings)

    # 新实现的方法
    def check_finished_recordings(self) -> List[Dict]:
        """
        在没有新帧的情况下检查并完成录制。
        这对于处理视频流结束时挂起的录制非常重要。
        """
        if not self.has_active_recordings() or not self.frame_ids:
            return []

        last_known_frame_id = self.frame_ids[-1]
        completed_recordings_info = []
        events_to_remove = []

        for event_time, recording_info in self.active_recordings.items():
            if last_known_frame_id >= recording_info['end_frame_id']:
                # 根据已知的最后一帧，此录制已完成
                final_info = self._finalize_recording(recording_info)
                completed_recordings_info.append(final_info)
                events_to_remove.append(event_time)

        for event_time in events_to_remove:
            del self.active_recordings[event_time]

        return completed_recordings_info
        
    def _finalize_recording(self, recording_info: Dict) -> Dict:
        """
        [辅助方法] 关闭视频写入器，创建元数据文件，并记录日志。
        """
        recording_info['video_writer'].release()
        self._create_metadata_file(recording_info)

        actual_duration = recording_info['frames_written'] / self.fps
        
        # 使用录制信息中的数据计算准确的 pre/post 时间
        event_frame_id = recording_info['event'].frame_id
        start_frame_id = recording_info['start_frame_id']
        # 实际的结束帧是最后写入的帧
        actual_end_frame_id = event_frame_id + (recording_info['frames_written'] - (event_frame_id - start_frame_id))

        pre_event_actual_frames = event_frame_id - start_frame_id
        post_event_actual_frames = actual_end_frame_id - event_frame_id
        
        logger.info(f"Completed recording: {recording_info['output_path']}")
        logger.info(f"  - Frames Written: {recording_info['frames_written']}, Total Duration: {actual_duration:.2f}s")
        logger.info(f"  - Pre-event: {pre_event_actual_frames} frames ({pre_event_actual_frames/self.fps:.2f}s)")
        logger.info(f"  - Post-event: {post_event_actual_frames} frames ({post_event_actual_frames/self.fps:.2f}s)")
        
        # 返回包含事件的完整信息，以便上游处理
        return recording_info

    def _create_metadata_file(self, recording_info: Dict):
        """创建元数据文件，包含准确的帧信息"""
        metadata_path = Path(recording_info['output_path']).with_suffix('.json')
        
        event = recording_info['event']
        start_frame_id = recording_info['start_frame_id']
        event_frame_id = event.frame_id
        # 实际写入的帧数决定了视频的长度
        frames_written_after_event = recording_info['frames_written'] - (event_frame_id - start_frame_id)
        
        pre_event_actual_frames = event_frame_id - start_frame_id
        post_event_actual_frames = max(0, frames_written_after_event) # 确保不为负

        metadata = {
            'event': {
                'timestamp': event.timestamp,
                'action_id': event.action_id,
                'action_name': event.action_name,
                'confidence': event.confidence,
                'enhanced': event.enhanced,
                'frame_id': event.frame_id
            },
            'recording': {
                'output_path': recording_info['output_path'],
                'frames_written': recording_info['frames_written'],
                'fps': self.fps,
                'start_frame_id': start_frame_id,
                'end_frame_id': start_frame_id + recording_info['frames_written'] -1, # 实际的最后一帧ID
                'pre_event_frames_actual': pre_event_actual_frames,
                'pre_event_seconds_actual': pre_event_actual_frames / self.fps,
                'post_event_frames_actual': post_event_actual_frames,
                'post_event_seconds_actual': post_event_actual_frames / self.fps
            },
            'created_at': datetime.now().isoformat()
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def stop_all_recordings(self):
        """立即停止并最终化所有活跃的录制任务。"""
        if not self.has_active_recordings():
            return
            
        logger.info(f"Stopping all {len(self.active_recordings)} active recordings...")
        for recording_info in self.active_recordings.values():
            self._finalize_recording(recording_info)

        self.active_recordings.clear()
        logger.info("All active recordings have been stopped.")