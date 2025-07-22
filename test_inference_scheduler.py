#!/usr/bin/env python3
"""
推理调度器测试脚本
验证新的推理策略的正确性和性能提升
"""

import time
import numpy as np
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from utils.inference_scheduler import InferenceScheduler, SamplingStrategy

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_mock_skeleton_data(num_frames: int = 1000) -> List[np.ndarray]:
    """生成模拟骨架数据用于测试"""
    mock_data = []
    for i in range(num_frames):
        # 生成25个关键点的模拟数据 (x, y, confidence)
        skeleton = np.random.rand(25, 3).astype(np.float32)
        skeleton[:, 2] = np.random.uniform(0.5, 1.0, 25)  # 置信度在0.5-1.0之间
        mock_data.append(skeleton)
    return mock_data


def test_sampling_strategies():
    """测试不同的采样策略"""
    logger.info("开始测试不同的采样策略...")
    
    # 测试参数
    window_size = 64
    num_frames = 500
    batch_size = 3
    
    # 生成测试数据
    mock_data = generate_mock_skeleton_data(num_frames)
    
    # 测试不同策略
    strategies = [
        (SamplingStrategy.SLIDING_WINDOW, 1),
        (SamplingStrategy.STRIDE_BASED, 3),
        (SamplingStrategy.STRIDE_BASED, window_size),
        (SamplingStrategy.ADAPTIVE, 3)
    ]
    
    results = {}
    
    for strategy, stride in strategies:
        logger.info(f"测试策略: {strategy.value}, 步幅: {stride}")
        
        # 创建调度器
        scheduler = InferenceScheduler(
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            strategy=strategy,
            max_buffer_size=200
        )
        
        # 记录性能指标
        start_time = time.time()
        total_batches = 0
        total_segments = 0
        
        # 模拟帧输入
        for i, frame_data in enumerate(mock_data):
            batch = scheduler.feed_frame(frame_data, i, time.time())
            if batch is not None:
                total_batches += 1
                total_segments += len(batch.segments)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 获取统计信息
        stats = scheduler.get_stats()
        
        results[f"{strategy.value}_stride_{stride}"] = {
            'strategy': strategy.value,
            'stride': stride,
            'processing_time': processing_time,
            'total_batches': total_batches,
            'total_segments': total_segments,
            'avg_batch_size': stats['avg_batch_size'],
            'buffer_utilization': stats['buffer_utilization'],
            'avg_processing_time': stats.get('avg_processing_time', 0),
            'frames_per_second': num_frames / processing_time if processing_time > 0 else 0,
            'segments_per_frame': total_segments / num_frames if num_frames > 0 else 0
        }
        
        logger.info(f"  处理时间: {processing_time:.3f}s")
        logger.info(f"  总批次数: {total_batches}")
        logger.info(f"  总片段数: {total_segments}")
        logger.info(f"  平均批次大小: {stats['avg_batch_size']:.2f}")
        logger.info(f"  处理速度: {num_frames / processing_time:.1f} FPS")
        logger.info(f"  片段/帧比率: {total_segments / num_frames:.3f}")
        logger.info("")
    
    return results


def test_batch_sizes():
    """测试不同批量大小的性能影响"""
    logger.info("开始测试不同批量大小的性能影响...")
    
    window_size = 64
    stride = 3
    num_frames = 300
    batch_sizes = [1, 2, 3, 5, 8]
    
    mock_data = generate_mock_skeleton_data(num_frames)
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"测试批量大小: {batch_size}")
        
        scheduler = InferenceScheduler(
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            strategy=SamplingStrategy.STRIDE_BASED,
            max_buffer_size=200
        )
        
        start_time = time.time()
        total_batches = 0
        
        for i, frame_data in enumerate(mock_data):
            batch = scheduler.feed_frame(frame_data, i, time.time())
            if batch is not None:
                total_batches += 1
        
        processing_time = time.time() - start_time
        stats = scheduler.get_stats()
        
        results[f"batch_size_{batch_size}"] = {
            'batch_size': batch_size,
            'processing_time': processing_time,
            'total_batches': total_batches,
            'avg_batch_size': stats['avg_batch_size'],
            'frames_per_second': num_frames / processing_time if processing_time > 0 else 0
        }
        
        logger.info(f"  处理时间: {processing_time:.3f}s")
        logger.info(f"  总批次数: {total_batches}")
        logger.info(f"  实际平均批次大小: {stats['avg_batch_size']:.2f}")
        logger.info(f"  处理速度: {num_frames / processing_time:.1f} FPS")
        logger.info("")
    
    return results


def test_adaptive_strategy():
    """测试自适应策略的动态调整能力"""
    logger.info("开始测试自适应策略...")
    
    window_size = 64
    num_frames = 400
    batch_size = 3
    
    # 自适应配置
    adaptive_config = {
        'min_stride': 1,
        'max_stride': 10,
        'load_threshold_high': 0.7,
        'load_threshold_low': 0.3,
        'adjustment_factor': 1.5
    }
    
    scheduler = InferenceScheduler(
        window_size=window_size,
        stride=3,
        batch_size=batch_size,
        strategy=SamplingStrategy.ADAPTIVE,
        max_buffer_size=100,  # 较小的缓冲区以触发自适应调整
        adaptive_config=adaptive_config
    )
    
    mock_data = generate_mock_skeleton_data(num_frames)
    
    # 记录缓冲区利用率变化
    buffer_utilizations = []
    segment_counts = []
    
    for i, frame_data in enumerate(mock_data):
        batch = scheduler.feed_frame(frame_data, i, time.time())
        
        stats = scheduler.get_stats()
        buffer_utilizations.append(stats['buffer_utilization'])
        segment_counts.append(stats['processed_segments'])
        
        if i % 50 == 0:
            logger.info(f"帧 {i}: 缓冲区利用率 {stats['buffer_utilization']:.2%}, "
                       f"已处理片段 {stats['processed_segments']}")
    
    final_stats = scheduler.get_stats()
    logger.info(f"自适应策略测试完成:")
    logger.info(f"  总处理片段: {final_stats['processed_segments']}")
    logger.info(f"  总处理批次: {final_stats['processed_batches']}")
    logger.info(f"  平均缓冲区利用率: {np.mean(buffer_utilizations):.2%}")
    
    return {
        'buffer_utilizations': buffer_utilizations,
        'segment_counts': segment_counts,
        'final_stats': final_stats
    }


def compare_with_traditional_approach():
    """与传统滑动窗口方法进行性能对比"""
    logger.info("开始与传统方法进行性能对比...")
    
    window_size = 64
    num_frames = 500
    mock_data = generate_mock_skeleton_data(num_frames)
    
    # 传统滑动窗口 (stride=1)
    traditional_scheduler = InferenceScheduler(
        window_size=window_size,
        stride=1,
        batch_size=1,
        strategy=SamplingStrategy.SLIDING_WINDOW
    )
    
    # 优化的步幅采样 (stride=3, batch_size=3)
    optimized_scheduler = InferenceScheduler(
        window_size=window_size,
        stride=3,
        batch_size=3,
        strategy=SamplingStrategy.STRIDE_BASED
    )
    
    # 测试传统方法
    start_time = time.time()
    traditional_batches = 0
    traditional_segments = 0
    
    for i, frame_data in enumerate(mock_data):
        batch = traditional_scheduler.feed_frame(frame_data, i, time.time())
        if batch is not None:
            traditional_batches += 1
            traditional_segments += len(batch.segments)
    
    traditional_time = time.time() - start_time
    traditional_stats = traditional_scheduler.get_stats()
    
    # 测试优化方法
    start_time = time.time()
    optimized_batches = 0
    optimized_segments = 0
    
    for i, frame_data in enumerate(mock_data):
        batch = optimized_scheduler.feed_frame(frame_data, i, time.time())
        if batch is not None:
            optimized_batches += 1
            optimized_segments += len(batch.segments)
    
    optimized_time = time.time() - start_time
    optimized_stats = optimized_scheduler.get_stats()
    
    # 计算性能提升
    time_improvement = (traditional_time - optimized_time) / traditional_time * 100
    segment_reduction = (traditional_segments - optimized_segments) / traditional_segments * 100
    
    logger.info("性能对比结果:")
    logger.info(f"传统方法:")
    logger.info(f"  处理时间: {traditional_time:.3f}s")
    logger.info(f"  总批次数: {traditional_batches}")
    logger.info(f"  总片段数: {traditional_segments}")
    logger.info(f"  处理速度: {num_frames / traditional_time:.1f} FPS")
    
    logger.info(f"优化方法:")
    logger.info(f"  处理时间: {optimized_time:.3f}s")
    logger.info(f"  总批次数: {optimized_batches}")
    logger.info(f"  总片段数: {optimized_segments}")
    logger.info(f"  处理速度: {num_frames / optimized_time:.1f} FPS")
    
    logger.info(f"性能提升:")
    logger.info(f"  时间减少: {time_improvement:.1f}%")
    logger.info(f"  片段减少: {segment_reduction:.1f}%")
    logger.info(f"  速度提升: {optimized_time / traditional_time:.2f}x")
    
    return {
        'traditional': {
            'time': traditional_time,
            'batches': traditional_batches,
            'segments': traditional_segments,
            'fps': num_frames / traditional_time
        },
        'optimized': {
            'time': optimized_time,
            'batches': optimized_batches,
            'segments': optimized_segments,
            'fps': num_frames / optimized_time
        },
        'improvements': {
            'time_reduction_percent': time_improvement,
            'segment_reduction_percent': segment_reduction,
            'speed_multiplier': traditional_time / optimized_time if optimized_time > 0 else 0
        }
    }


def main():
    """主测试函数"""
    logger.info("开始推理调度器测试...")
    
    # 测试1: 不同采样策略
    strategy_results = test_sampling_strategies()
    
    # 测试2: 不同批量大小
    batch_results = test_batch_sizes()
    
    # 测试3: 自适应策略
    adaptive_results = test_adaptive_strategy()
    
    # 测试4: 性能对比
    comparison_results = compare_with_traditional_approach()
    
    logger.info("所有测试完成!")
    
    # 输出总结
    logger.info("\n" + "="*50)
    logger.info("测试总结:")
    logger.info("="*50)
    
    logger.info("1. 采样策略测试:")
    for key, result in strategy_results.items():
        logger.info(f"   {key}: {result['frames_per_second']:.1f} FPS, "
                   f"片段/帧比率: {result['segments_per_frame']:.3f}")
    
    logger.info("2. 批量大小测试:")
    for key, result in batch_results.items():
        logger.info(f"   {key}: {result['frames_per_second']:.1f} FPS, "
                   f"批次数: {result['total_batches']}")
    
    logger.info("3. 性能对比:")
    comp = comparison_results['improvements']
    logger.info(f"   时间减少: {comp['time_reduction_percent']:.1f}%")
    logger.info(f"   片段减少: {comp['segment_reduction_percent']:.1f}%")
    logger.info(f"   速度提升: {comp['speed_multiplier']:.2f}x")


if __name__ == "__main__":
    main()
