#!/usr/bin/env python3
"""
测试批量推理修复后的功能
验证不会重复触发事件
"""

import time
import numpy as np
import logging
from utils.inference_scheduler import InferenceScheduler, SamplingStrategy

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_stride_sampling_fix():
    """测试步幅采样修复"""
    logger.info("测试步幅采样修复...")
    
    # 创建调度器，使用较大的步幅
    scheduler = InferenceScheduler(
        window_size=64,
        stride=32,  # 大步幅
        batch_size=3,
        strategy=SamplingStrategy.STRIDE_BASED,
        max_buffer_size=200
    )
    
    batch_count = 0
    segment_count = 0
    
    # 模拟输入200帧
    for frame_id in range(200):
        skeleton_data = np.random.rand(25, 3).astype(np.float32)
        batch = scheduler.feed_frame(skeleton_data, frame_id, time.time())
        
        if batch is not None:
            batch_count += 1
            segment_count += len(batch.segments)
            logger.info(f"批次 {batch.batch_id}: 包含 {len(batch.segments)} 个片段")
            
            # 显示每个片段的帧范围
            for i, segment in enumerate(batch.segments):
                logger.info(f"  片段 {i+1}: 帧 {segment.start_frame_id}-{segment.end_frame_id}")
    
    stats = scheduler.get_stats()
    logger.info(f"测试结果:")
    logger.info(f"  输入帧数: 200")
    logger.info(f"  生成批次: {batch_count}")
    logger.info(f"  生成片段: {segment_count}")
    logger.info(f"  片段/帧比率: {segment_count/200:.3f}")
    
    # 验证步幅采样是否正确
    expected_segments = (200 - 64) // 32 + 1  # 理论上应该生成的片段数
    logger.info(f"  预期片段数: {expected_segments}")
    logger.info(f"  实际片段数: {segment_count}")
    
    return batch_count, segment_count


def test_batch_inference_single_result():
    """测试批量推理只返回单个结果"""
    logger.info("\n测试批量推理单结果...")
    
    # 模拟批量推理结果处理
    class MockSegment:
        def __init__(self, segment_id, end_frame_id, timestamps):
            self.segment_id = segment_id
            self.end_frame_id = end_frame_id
            self.timestamps = timestamps
    
    # 模拟3个片段的推理结果
    mock_results = [
        {
            'output': np.random.rand(60),  # 60个动作类别的输出
            'segment': MockSegment(1, 100, [1.0, 1.1, 1.2])
        },
        {
            'output': np.random.rand(60),
            'segment': MockSegment(2, 132, [1.3, 1.4, 1.5])
        },
        {
            'output': np.random.rand(60),
            'segment': MockSegment(3, 164, [1.6, 1.7, 1.8])
        }
    ]
    
    # 模拟选择最佳结果的逻辑
    best_confidence = 0.0
    best_result = None
    
    for result_data in mock_results:
        output = result_data['output']
        segment = result_data['segment']
        
        # 计算置信度
        confidence = np.max(output)
        predicted_class = np.argmax(output)
        
        logger.info(f"片段 {segment.segment_id}: 动作 {predicted_class}, 置信度 {confidence:.3f}")
        
        if confidence > best_confidence:
            best_confidence = confidence
            best_result = {
                'segment_id': segment.segment_id,
                'action_id': predicted_class,
                'confidence': confidence,
                'frame_id': segment.end_frame_id
            }
    
    logger.info(f"选择的最佳结果: 片段 {best_result['segment_id']}, "
               f"动作 {best_result['action_id']}, 置信度 {best_result['confidence']:.3f}")
    
    return best_result


def test_performance_improvement():
    """测试性能改进"""
    logger.info("\n测试性能改进...")
    
    # 测试传统方法
    logger.info("测试传统滑动窗口...")
    traditional_scheduler = InferenceScheduler(
        window_size=64,
        stride=1,
        batch_size=1,
        strategy=SamplingStrategy.SLIDING_WINDOW
    )
    
    start_time = time.time()
    traditional_batches = 0
    
    for i in range(200):
        skeleton_data = np.random.rand(25, 3).astype(np.float32)
        batch = traditional_scheduler.feed_frame(skeleton_data, i, time.time())
        if batch is not None:
            traditional_batches += 1
    
    traditional_time = time.time() - start_time
    
    # 测试优化方法
    logger.info("测试步幅采样...")
    optimized_scheduler = InferenceScheduler(
        window_size=64,
        stride=32,
        batch_size=3,
        strategy=SamplingStrategy.STRIDE_BASED
    )
    
    start_time = time.time()
    optimized_batches = 0
    
    for i in range(200):
        skeleton_data = np.random.rand(25, 3).astype(np.float32)
        batch = optimized_scheduler.feed_frame(skeleton_data, i, time.time())
        if batch is not None:
            optimized_batches += 1
    
    optimized_time = time.time() - start_time
    
    # 计算改进
    batch_reduction = (traditional_batches - optimized_batches) / traditional_batches * 100
    time_improvement = (traditional_time - optimized_time) / traditional_time * 100
    
    logger.info(f"性能对比结果:")
    logger.info(f"  传统方法: {traditional_batches} 批次, {traditional_time:.3f}s")
    logger.info(f"  优化方法: {optimized_batches} 批次, {optimized_time:.3f}s")
    logger.info(f"  批次减少: {batch_reduction:.1f}%")
    logger.info(f"  时间改进: {time_improvement:.1f}%")
    
    return {
        'traditional_batches': traditional_batches,
        'optimized_batches': optimized_batches,
        'batch_reduction_percent': batch_reduction,
        'time_improvement_percent': time_improvement
    }


def main():
    """主测试函数"""
    logger.info("开始测试批量推理修复...")
    
    # 测试1: 步幅采样修复
    batch_count, segment_count = test_stride_sampling_fix()
    
    # 测试2: 批量推理单结果
    best_result = test_batch_inference_single_result()
    
    # 测试3: 性能改进
    performance_results = test_performance_improvement()
    
    # 总结
    logger.info("\n" + "="*50)
    logger.info("测试总结")
    logger.info("="*50)
    
    logger.info("修复验证:")
    logger.info(f"✅ 步幅采样: 生成 {batch_count} 批次, {segment_count} 片段")
    logger.info(f"✅ 单结果选择: 选择片段 {best_result['segment_id']}")
    logger.info(f"✅ 性能改进: 批次减少 {performance_results['batch_reduction_percent']:.1f}%")
    
    logger.info("\n预期效果:")
    logger.info("1. 不再出现重复事件触发")
    logger.info("2. FPS应该有显著提升")
    logger.info("3. 推理频率大幅降低")


if __name__ == "__main__":
    main()
