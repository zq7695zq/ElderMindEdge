"""
视频流推理调度器模块
实现步幅采样、批量推理等优化策略，提升推理效率，降低冗余计算
"""

import time
import logging
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any, Callable, Union
from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """采样策略枚举"""
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口模式 (stride=1)
    STRIDE_BASED = "stride_based"      # 步幅采样模式 (stride=window_size)
    ADAPTIVE = "adaptive"              # 自适应采样模式
    CUSTOM = "custom"                  # 自定义采样策略


@dataclass
class WindowSegment:
    """窗口片段数据结构"""
    frames: List[np.ndarray]  # 帧数据列表
    frame_ids: List[int]      # 帧ID列表
    timestamps: List[float]   # 时间戳列表
    start_frame_id: int       # 起始帧ID
    end_frame_id: int         # 结束帧ID
    segment_id: int           # 片段ID


@dataclass
class InferenceBatch:
    """推理批次数据结构"""
    segments: List[WindowSegment]  # 窗口片段列表
    batch_id: int                  # 批次ID
    created_time: float            # 创建时间


class InferenceScheduler:
    """推理调度器 - 负责采样策略和批量推理调度"""
    
    def __init__(self, 
                 window_size: int = 64,
                 stride: int = 1,
                 batch_size: int = 1,
                 strategy: SamplingStrategy = SamplingStrategy.SLIDING_WINDOW,
                 max_buffer_size: int = 200,
                 adaptive_config: Optional[Dict[str, Any]] = None):
        """
        初始化推理调度器
        
        Args:
            window_size: 窗口大小
            stride: 步幅大小
            batch_size: 批量大小
            strategy: 采样策略
            max_buffer_size: 最大缓冲区大小
            adaptive_config: 自适应配置参数
        """
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.strategy = strategy
        self.max_buffer_size = max_buffer_size
        
        # 验证参数
        if stride > window_size:
            logger.warning(f"步幅 {stride} 大于窗口大小 {window_size}，可能导致关键帧丢失")
        
        # 帧缓冲区
        self.frame_buffer: deque = deque(maxlen=max_buffer_size)
        
        # 待处理的窗口片段队列
        self.pending_segments: deque = deque()

        # 批次管理
        self.current_batch: List[WindowSegment] = []
        self.batch_counter = 0
        self.segment_counter = 0

        # 步幅采样状态跟踪
        self.last_processed_frame_id = -1
        
        # 自适应配置
        self.adaptive_config = adaptive_config or {
            'min_stride': 1,
            'max_stride': window_size,
            'load_threshold_high': 0.8,
            'load_threshold_low': 0.3,
            'adjustment_factor': 1.2
        }
        
        # 性能统计
        self.stats = {
            'total_frames': 0,
            'processed_segments': 0,
            'processed_batches': 0,
            'avg_batch_size': 0.0,
            'buffer_utilization': 0.0,
            'processing_times': deque(maxlen=100)
        }
        
        # 线程安全锁
        self.lock = threading.Lock()
        
        logger.info(f"推理调度器初始化: window_size={window_size}, stride={stride}, "
                   f"batch_size={batch_size}, strategy={strategy.value}")
    
    def feed_frame(self, frame_data: np.ndarray, frame_id: int, timestamp: float) -> Optional[InferenceBatch]:
        """
        输入单帧数据，自动判断是否触发批次输出
        
        Args:
            frame_data: 帧数据 (骨架数据)
            frame_id: 帧ID
            timestamp: 时间戳
            
        Returns:
            如果触发批次输出，返回 InferenceBatch，否则返回 None
        """
        with self.lock:
            start_time = time.time()
            
            # 添加帧到缓冲区
            self.frame_buffer.append({
                'data': frame_data,
                'frame_id': frame_id,
                'timestamp': timestamp
            })
            self.stats['total_frames'] += 1
            
            # 根据策略进行采样
            new_segments = self._sample_segments()
            
            # 将新片段添加到待处理队列
            for segment in new_segments:
                self.pending_segments.append(segment)
            
            # 尝试构建批次
            batch = self._try_build_batch()
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.stats['buffer_utilization'] = len(self.frame_buffer) / self.max_buffer_size
            
            return batch
    
    def _sample_segments(self) -> List[WindowSegment]:
        """根据当前策略进行采样，返回新的窗口片段"""
        if len(self.frame_buffer) < self.window_size:
            return []
        
        segments = []
        
        if self.strategy == SamplingStrategy.SLIDING_WINDOW:
            segments = self._sliding_window_sampling()
        elif self.strategy == SamplingStrategy.STRIDE_BASED:
            segments = self._stride_based_sampling()
        elif self.strategy == SamplingStrategy.ADAPTIVE:
            segments = self._adaptive_sampling()
        elif self.strategy == SamplingStrategy.CUSTOM:
            segments = self._custom_sampling()
        
        return segments
    
    def _sliding_window_sampling(self) -> List[WindowSegment]:
        """滑动窗口采样 (stride=1)"""
        segments = []
        
        # 每次新帧都创建一个新的窗口片段
        if len(self.frame_buffer) >= self.window_size:
            frames = list(self.frame_buffer)[-self.window_size:]
            segment = self._create_segment(frames)
            segments.append(segment)
        
        return segments
    
    def _stride_based_sampling(self) -> List[WindowSegment]:
        """步幅采样 - 只处理新的片段"""
        segments = []

        if len(self.frame_buffer) < self.window_size:
            return segments

        buffer_frames = list(self.frame_buffer)
        current_frame_id = buffer_frames[-1]['frame_id']

        # 计算下一个应该处理的起始帧ID
        next_start_frame_id = self.last_processed_frame_id + self.stride

        # 检查是否可以创建新的完整片段
        for i, frame_data in enumerate(buffer_frames):
            if frame_data['frame_id'] >= next_start_frame_id:
                # 检查是否有足够的帧来创建完整窗口
                if i + self.window_size <= len(buffer_frames):
                    frames = buffer_frames[i:i + self.window_size]
                    segment = self._create_segment(frames)
                    segments.append(segment)

                    # 更新最后处理的帧ID
                    self.last_processed_frame_id = frames[0]['frame_id']
                    break

        return segments
    
    def _adaptive_sampling(self) -> List[WindowSegment]:
        """自适应采样 - 根据系统负载动态调整步幅"""
        # 根据缓冲区利用率调整步幅
        current_utilization = len(self.frame_buffer) / self.max_buffer_size
        
        if current_utilization > self.adaptive_config['load_threshold_high']:
            # 高负载：增加步幅，减少采样频率
            adaptive_stride = min(
                int(self.stride * self.adaptive_config['adjustment_factor']),
                self.adaptive_config['max_stride']
            )
        elif current_utilization < self.adaptive_config['load_threshold_low']:
            # 低负载：减少步幅，增加采样频率
            adaptive_stride = max(
                int(self.stride / self.adaptive_config['adjustment_factor']),
                self.adaptive_config['min_stride']
            )
        else:
            adaptive_stride = self.stride
        
        # 使用调整后的步幅进行采样
        original_stride = self.stride
        self.stride = adaptive_stride
        segments = self._stride_based_sampling()
        self.stride = original_stride  # 恢复原始步幅
        
        return segments
    
    def _custom_sampling(self) -> List[WindowSegment]:
        """自定义采样策略 - 可由子类重写"""
        # 默认使用步幅采样
        return self._stride_based_sampling()
    
    def _create_segment(self, frames: List[Dict]) -> WindowSegment:
        """创建窗口片段"""
        self.segment_counter += 1
        
        frame_data = [f['data'] for f in frames]
        frame_ids = [f['frame_id'] for f in frames]
        timestamps = [f['timestamp'] for f in frames]
        
        return WindowSegment(
            frames=frame_data,
            frame_ids=frame_ids,
            timestamps=timestamps,
            start_frame_id=frame_ids[0],
            end_frame_id=frame_ids[-1],
            segment_id=self.segment_counter
        )
    
    def _try_build_batch(self) -> Optional[InferenceBatch]:
        """尝试构建推理批次"""
        if len(self.pending_segments) >= self.batch_size:
            # 收集足够的片段构建批次
            batch_segments = []
            for _ in range(self.batch_size):
                if self.pending_segments:
                    batch_segments.append(self.pending_segments.popleft())
            
            if batch_segments:
                self.batch_counter += 1
                batch = InferenceBatch(
                    segments=batch_segments,
                    batch_id=self.batch_counter,
                    created_time=time.time()
                )
                
                # 更新统计
                self.stats['processed_batches'] += 1
                self.stats['processed_segments'] += len(batch_segments)
                self.stats['avg_batch_size'] = (
                    self.stats['avg_batch_size'] * (self.stats['processed_batches'] - 1) + 
                    len(batch_segments)
                ) / self.stats['processed_batches']
                
                return batch
        
        return None
    
    def get_pending_count(self) -> int:
        """获取待处理片段数量"""
        with self.lock:
            return len(self.pending_segments)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        with self.lock:
            stats_copy = self.stats.copy()
            stats_copy['avg_processing_time'] = (
                np.mean(self.stats['processing_times']) 
                if self.stats['processing_times'] else 0.0
            )
            stats_copy['current_buffer_size'] = len(self.frame_buffer)
            stats_copy['pending_segments'] = len(self.pending_segments)
            
            return stats_copy
    
    def reset_stats(self):
        """重置统计信息"""
        with self.lock:
            self.stats = {
                'total_frames': 0,
                'processed_segments': 0,
                'processed_batches': 0,
                'avg_batch_size': 0.0,
                'buffer_utilization': 0.0,
                'processing_times': deque(maxlen=100)
            }
    
    def clear_buffers(self):
        """清空所有缓冲区"""
        with self.lock:
            self.frame_buffer.clear()
            self.pending_segments.clear()
            self.current_batch.clear()
            self.last_processed_frame_id = -1
    
    def update_config(self, **kwargs):
        """动态更新配置参数"""
        with self.lock:
            if 'window_size' in kwargs:
                self.window_size = kwargs['window_size']
            if 'stride' in kwargs:
                self.stride = kwargs['stride']
                if self.stride > self.window_size:
                    logger.warning(f"步幅 {self.stride} 大于窗口大小 {self.window_size}")
            if 'batch_size' in kwargs:
                self.batch_size = kwargs['batch_size']
            if 'strategy' in kwargs:
                self.strategy = kwargs['strategy']
            if 'adaptive_config' in kwargs:
                self.adaptive_config.update(kwargs['adaptive_config'])
            
            logger.info(f"推理调度器配置已更新: {kwargs}")


class BatchInferenceProcessor:
    """批量推理处理器 - 负责将批次数据转换为模型输入并处理推理结果"""

    def __init__(self, preprocessor, device: torch.device):
        """
        初始化批量推理处理器

        Args:
            preprocessor: 数据预处理器 (SkateFormerPreprocessor)
            device: 推理设备
        """
        self.preprocessor = preprocessor
        self.device = device
        self.processing_stats = {
            'total_batches': 0,
            'total_segments': 0,
            'avg_batch_processing_time': 0.0,
            'processing_times': deque(maxlen=100)
        }

    def process_batch(self, batch: InferenceBatch) -> List[Tuple[torch.Tensor, torch.Tensor, WindowSegment]]:
        """
        处理推理批次，将窗口片段转换为模型输入

        Args:
            batch: 推理批次

        Returns:
            List of (data_tensor, index_tensor, segment) tuples
        """
        start_time = time.time()
        results = []

        for segment in batch.segments:
            try:
                # 为每个片段创建临时预处理器实例或重置现有实例
                temp_preprocessor = self._create_temp_preprocessor()

                # 逐帧添加到预处理器
                preprocessed_data = None
                for frame_data in segment.frames:
                    preprocessed_data = temp_preprocessor.add_frame(frame_data)

                if preprocessed_data is not None:
                    data_tensor, index_tensor = preprocessed_data
                    # 确保张量在正确的设备上
                    data_tensor = data_tensor.to(self.device, non_blocking=True)
                    index_tensor = index_tensor.to(self.device, non_blocking=True)

                    results.append((data_tensor, index_tensor, segment))
                else:
                    logger.warning(f"片段 {segment.segment_id} 预处理失败")

            except Exception as e:
                import traceback
                logger.error(f"处理片段 {segment.segment_id} 时出错: {e}\n{traceback.format_exc()}")
                continue

        # 更新统计信息
        processing_time = time.time() - start_time
        self.processing_stats['processing_times'].append(processing_time)
        self.processing_stats['total_batches'] += 1
        self.processing_stats['total_segments'] += len(batch.segments)
        self.processing_stats['avg_batch_processing_time'] = (
            np.mean(self.processing_stats['processing_times'])
            if self.processing_stats['processing_times'] else 0.0
        )

        logger.debug(f"批次 {batch.batch_id} 处理完成: {len(results)}/{len(batch.segments)} 个片段成功")
        return results

    def _create_temp_preprocessor(self):
        """创建临时预处理器实例"""
        # 创建一个新的预处理器实例，避免状态污染
        from utils.preprocessor import SkateFormerPreprocessor
        return SkateFormerPreprocessor(
            window_size=self.preprocessor.window_size,
            num_people=self.preprocessor.num_people,
            num_points=self.preprocessor.num_points
        )

    def batch_inference(self, model, preprocessed_batch: List[Tuple[torch.Tensor, torch.Tensor, WindowSegment]]) -> List[Dict[str, Any]]:
        """
        执行批量推理

        Args:
            model: SkateFormer模型
            preprocessed_batch: 预处理后的批次数据

        Returns:
            推理结果列表
        """
        if not preprocessed_batch:
            return []

        start_time = time.time()
        results = []

        try:
            # 提取所有数据张量和索引张量
            data_tensors = []
            index_tensors = []
            segments = []

            for data_tensor, index_tensor, segment in preprocessed_batch:
                data_tensors.append(data_tensor)
                index_tensors.append(index_tensor)
                segments.append(segment)

            if len(data_tensors) == 1:
                # 单个样本推理
                batch_data = data_tensors[0]
                batch_indices = index_tensors[0]
            else:
                # 多个样本批量推理
                batch_data = torch.cat(data_tensors, dim=0)
                batch_indices = torch.cat(index_tensors, dim=0)

            # 执行推理
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                with torch.no_grad():
                    outputs = model(batch_data, batch_indices)

            # 处理输出结果
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)

            for i, (output, segment) in enumerate(zip(outputs, segments)):
                result = {
                    'output': output,
                    'segment': segment,
                    'inference_time': time.time() - start_time,
                    'batch_size': len(segments)
                }
                results.append(result)

        except Exception as e:
            logger.error(f"批量推理失败: {e}")
            # 如果批量推理失败，尝试单个推理
            for data_tensor, index_tensor, segment in preprocessed_batch:
                try:
                    with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                        with torch.no_grad():
                            output = model(data_tensor, index_tensor)

                    result = {
                        'output': output.squeeze(0) if output.dim() > 1 else output,
                        'segment': segment,
                        'inference_time': time.time() - start_time,
                        'batch_size': 1
                    }
                    results.append(result)
                except Exception as single_e:
                    logger.error(f"单个推理也失败 (片段 {segment.segment_id}): {single_e}")

        inference_time = time.time() - start_time
        logger.debug(f"批量推理完成: {len(results)} 个结果, 耗时 {inference_time:.3f}s")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.processing_stats.copy()


def create_inference_scheduler(config: Dict[str, Any]) -> InferenceScheduler:
    """
    根据配置创建推理调度器实例

    Args:
        config: 配置字典

    Returns:
        InferenceScheduler实例
    """
    scheduler_config = config.get('inference_scheduler', {})

    # 解析策略
    strategy_str = scheduler_config.get('strategy', 'sliding_window')
    try:
        strategy = SamplingStrategy(strategy_str)
    except ValueError:
        logger.warning(f"未知的采样策略: {strategy_str}, 使用默认策略")
        strategy = SamplingStrategy.SLIDING_WINDOW

    # 创建调度器
    scheduler = InferenceScheduler(
        window_size=scheduler_config.get('window_size', 64),
        stride=scheduler_config.get('stride', 1),
        batch_size=scheduler_config.get('batch_size', 1),
        strategy=strategy,
        max_buffer_size=scheduler_config.get('max_buffer_size', 200),
        adaptive_config=scheduler_config.get('adaptive_config')
    )

    return scheduler
