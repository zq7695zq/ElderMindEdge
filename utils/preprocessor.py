from utils.utils import apply_partition
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Deque
from dataclasses import dataclass
from collections import deque
import time


class SkateFormerPreprocessor:
    """Optimized SkateFormer data preprocessing with memory efficiency"""

    def __init__(self, window_size: int = 64, num_people: int = 2, buffer_size: int = None, num_points: int = 25):
        self.window_size = window_size
        self.num_people = num_people
        self.num_points = num_points

        # 使用固定大小的双端队列作为循环缓冲区
        buffer_size = buffer_size or (window_size * 2)
        self.data_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)

        # 预分配内存
        self.data_numpy = np.zeros((3, window_size, num_points, num_people), dtype=np.float32)
        self.temp_frames = np.zeros((window_size, num_points, 3), dtype=np.float32)
        self.indices = np.arange(window_size, dtype=np.float32)  # 直接使用float32类型

        # 缓存张量以减少内存分配
        self.cached_data_tensor = None
        self.cached_index_tensor = None

        # 性能统计
        self.processing_times = []

    def add_frame(self, skeleton_data: np.ndarray) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """添加帧数据并返回预处理结果（如果窗口已满）"""
        start_time = time.time()

        # 处理空数据
        if skeleton_data is None:
            skeleton_data = np.zeros((self.num_points, 3), dtype=np.float32)

        # 确保数据类型和形状正确
        if skeleton_data.dtype != np.float32:
            skeleton_data = skeleton_data.astype(np.float32)

        # 使用deque的自动大小管理
        self.data_buffer.append(skeleton_data)

        # 检查是否有足够的帧进行处理
        if len(self.data_buffer) >= self.window_size:
            result = self._preprocess_batch()

            # 记录处理时间
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:  # 保持最近100次的统计
                self.processing_times.pop(0)

            return result

        return None

    def _preprocess_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """优化的批处理预处理"""
        # 获取最近的window_size帧
        frames = list(self.data_buffer)[-self.window_size:]

        # 重置数组（比fill(0)更快）
        self.data_numpy[:] = 0
        self.temp_frames[:] = 0

        # 批量复制数据，减少循环开销
        for t, frame_data in enumerate(frames):
            if frame_data is not None and frame_data.size > 0:
                rows = min(self.num_points, frame_data.shape[0])
                cols = min(3, frame_data.shape[1])
                self.temp_frames[t, :rows, :cols] = frame_data[:rows, :cols]

        # 使用numpy的高效转置操作
        self.data_numpy[:, :self.window_size, :, 0] = self.temp_frames.transpose(2, 0, 1)

        # 归一化和分区
        self._normalize_coordinates()
        partitioned_data = self._apply_partition()

        # 重用张量以减少内存分配
        if self.cached_data_tensor is None or self.cached_data_tensor.shape[1:] != partitioned_data.shape:
            self.cached_data_tensor = torch.from_numpy(partitioned_data).float().unsqueeze(0)
            self.cached_index_tensor = torch.from_numpy(self.indices).float().unsqueeze(0)
        else:
            # 直接更新张量数据
            self.cached_data_tensor.data = torch.from_numpy(partitioned_data).float().unsqueeze(0).data

        return self.cached_data_tensor.clone(), self.cached_index_tensor.clone()

    def _normalize_coordinates(self):
        for c in range(2):
            channel_data = self.data_numpy[c]
            non_zero_mask = channel_data != 0

            if non_zero_mask.any():
                min_val = channel_data[non_zero_mask].min()
                max_val = channel_data[non_zero_mask].max()

                if max_val > min_val:
                    channel_data[non_zero_mask] = (channel_data[non_zero_mask] - min_val) / (max_val - min_val)
                    channel_data[non_zero_mask] = channel_data[non_zero_mask] * 2 - 1

    def _apply_partition(self) -> np.ndarray:
        # 只对 NTU 格式（25个关键点）应用分区
        if self.num_points == 25:
            return apply_partition(self.data_numpy)
        else:
            # 对于 YOLO pose 格式（17个关键点），直接返回原始数据
            return self.data_numpy

    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        if not self.processing_times:
            return {
                'avg_processing_time': 0.0,
                'max_processing_time': 0.0,
                'min_processing_time': 0.0,
                'buffer_size': len(self.data_buffer),
                'buffer_utilization': 0.0
            }

        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'buffer_size': len(self.data_buffer),
            'buffer_utilization': len(self.data_buffer) / self.data_buffer.maxlen if self.data_buffer.maxlen else 0.0
        }

    def reset_stats(self):
        """重置性能统计"""
        self.processing_times.clear()

    def clear_buffer(self):
        """清空缓冲区"""
        self.data_buffer.clear()
        self.data_numpy[:] = 0
        self.temp_frames[:] = 0