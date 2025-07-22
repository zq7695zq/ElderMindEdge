#!/usr/bin/env python3
"""
真实推理场景测试脚本
测试推理调度器在实际视频流处理中的性能表现
"""

import time
import logging
import numpy as np
import torch
from action_recognition_stream import ActionRecognitionStream
from utils.inference_scheduler import SamplingStrategy

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_inference_scheduler_integration():
    """测试推理调度器与主流处理器的集成"""
    logger.info("开始测试推理调度器集成...")
    
    # 测试配置
    test_configs = [
        {
            'name': '传统滑动窗口',
            'config': {
                'inference_scheduler': {
                    'enabled': False
                }
            }
        },
        {
            'name': '步幅采样 (stride=3, batch=1)',
            'config': {
                'inference_scheduler': {
                    'enabled': True,
                    'strategy': 'stride_based',
                    'window_size': 64,
                    'stride': 3,
                    'batch_size': 1
                }
            }
        },
        {
            'name': '步幅采样 (stride=3, batch=3)',
            'config': {
                'inference_scheduler': {
                    'enabled': True,
                    'strategy': 'stride_based',
                    'window_size': 64,
                    'stride': 3,
                    'batch_size': 3
                }
            }
        },
        {
            'name': '自适应采样',
            'config': {
                'inference_scheduler': {
                    'enabled': True,
                    'strategy': 'adaptive',
                    'window_size': 64,
                    'stride': 3,
                    'batch_size': 3,
                    'adaptive_config': {
                        'min_stride': 1,
                        'max_stride': 8,
                        'load_threshold_high': 0.8,
                        'load_threshold_low': 0.3,
                        'adjustment_factor': 1.2
                    }
                }
            }
        }
    ]
    
    results = {}
    
    for test_config in test_configs:
        logger.info(f"\n测试配置: {test_config['name']}")
        logger.info("-" * 50)
        
        try:
            # 创建动作识别流实例
            stream = ActionRecognitionStream(
                config_path=None,  # 使用默认配置
                enable_video_recording=False,  # 禁用视频录制以专注于推理性能
                event_callback=None
            )
            
            # 更新配置
            stream.config['stream_config'].update(test_config['config'])
            
            # 重新初始化推理调度器
            if test_config['config'].get('inference_scheduler', {}).get('enabled', False):
                from utils.inference_scheduler import create_inference_scheduler, BatchInferenceProcessor
                stream.use_inference_scheduler = True
                stream.inference_scheduler = create_inference_scheduler(stream.config['stream_config'])
                stream.batch_processor = BatchInferenceProcessor(stream.preprocessor, stream.device)
                logger.info(f"推理调度器已启用: 策略={stream.inference_scheduler.strategy.value}")
            else:
                stream.use_inference_scheduler = False
                stream.inference_scheduler = None
                stream.batch_processor = None
                logger.info("使用传统滑动窗口推理模式")
            
            # 模拟视频流处理
            start_time = time.time()
            
            # 生成模拟骨架数据
            num_frames = 200
            inference_count = 0
            
            for i in range(num_frames):
                # 生成模拟骨架数据
                skeleton_data = np.random.rand(25, 3).astype(np.float32)
                skeleton_data[:, 2] = np.random.uniform(0.5, 1.0, 25)  # 置信度
                
                if stream.use_inference_scheduler:
                    # 使用推理调度器
                    batch = stream.inference_scheduler.feed_frame(
                        skeleton_data, i, time.time()
                    )
                    if batch is not None:
                        inference_count += len(batch.segments)
                else:
                    # 使用传统方法
                    preprocessed_data = stream.preprocessor.add_frame(skeleton_data)
                    if preprocessed_data is not None:
                        inference_count += 1
            
            processing_time = time.time() - start_time
            
            # 收集统计信息
            stats = {
                'processing_time': processing_time,
                'frames_processed': num_frames,
                'inferences_triggered': inference_count,
                'fps': num_frames / processing_time if processing_time > 0 else 0,
                'inference_ratio': inference_count / num_frames if num_frames > 0 else 0
            }
            
            if stream.use_inference_scheduler:
                scheduler_stats = stream.inference_scheduler.get_stats()
                stats.update({
                    'scheduler_stats': scheduler_stats,
                    'batch_processor_stats': stream.batch_processor.get_stats() if stream.batch_processor else {}
                })
            
            results[test_config['name']] = stats
            
            # 输出结果
            logger.info(f"处理时间: {processing_time:.3f}s")
            logger.info(f"处理帧数: {num_frames}")
            logger.info(f"触发推理次数: {inference_count}")
            logger.info(f"处理速度: {stats['fps']:.1f} FPS")
            logger.info(f"推理比率: {stats['inference_ratio']:.3f}")
            
            if stream.use_inference_scheduler:
                logger.info(f"调度器统计:")
                logger.info(f"  总片段数: {scheduler_stats.get('processed_segments', 0)}")
                logger.info(f"  总批次数: {scheduler_stats.get('processed_batches', 0)}")
                logger.info(f"  平均批次大小: {scheduler_stats.get('avg_batch_size', 0):.2f}")
                logger.info(f"  缓冲区利用率: {scheduler_stats.get('buffer_utilization', 0):.2%}")
            
        except Exception as e:
            logger.error(f"测试配置 '{test_config['name']}' 失败: {e}")
            results[test_config['name']] = {'error': str(e)}
    
    return results


def compare_inference_strategies():
    """比较不同推理策略的性能"""
    logger.info("\n" + "="*60)
    logger.info("推理策略性能对比")
    logger.info("="*60)
    
    results = test_inference_scheduler_integration()
    
    # 分析结果
    if len(results) > 1:
        baseline_name = '传统滑动窗口'
        if baseline_name in results and 'error' not in results[baseline_name]:
            baseline = results[baseline_name]
            
            logger.info(f"\n基准测试 ({baseline_name}):")
            logger.info(f"  处理速度: {baseline['fps']:.1f} FPS")
            logger.info(f"  推理次数: {baseline['inferences_triggered']}")
            logger.info(f"  推理比率: {baseline['inference_ratio']:.3f}")
            
            logger.info(f"\n性能对比:")
            for name, result in results.items():
                if name != baseline_name and 'error' not in result:
                    speed_improvement = (result['fps'] - baseline['fps']) / baseline['fps'] * 100
                    inference_reduction = (baseline['inferences_triggered'] - result['inferences_triggered']) / baseline['inferences_triggered'] * 100
                    
                    logger.info(f"  {name}:")
                    logger.info(f"    速度变化: {speed_improvement:+.1f}%")
                    logger.info(f"    推理减少: {inference_reduction:+.1f}%")
                    logger.info(f"    推理比率: {result['inference_ratio']:.3f}")
    
    return results


def test_memory_efficiency():
    """测试内存使用效率"""
    logger.info("\n" + "="*60)
    logger.info("内存效率测试")
    logger.info("="*60)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # 测试传统方法
    logger.info("测试传统滑动窗口方法...")
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    stream_traditional = ActionRecognitionStream(
        enable_video_recording=False,
        event_callback=None
    )
    stream_traditional.use_inference_scheduler = False
    
    # 处理大量帧
    for i in range(1000):
        skeleton_data = np.random.rand(25, 3).astype(np.float32)
        stream_traditional.preprocessor.add_frame(skeleton_data)
    
    traditional_memory = process.memory_info().rss / 1024 / 1024  # MB
    traditional_usage = traditional_memory - initial_memory
    
    logger.info(f"传统方法内存使用: {traditional_usage:.1f} MB")
    
    # 测试优化方法
    logger.info("测试推理调度器方法...")
    
    # 重置内存基准
    del stream_traditional
    import gc
    gc.collect()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    stream_optimized = ActionRecognitionStream(
        enable_video_recording=False,
        event_callback=None
    )
    
    # 启用推理调度器
    stream_optimized.config['stream_config']['inference_scheduler'] = {
        'enabled': True,
        'strategy': 'stride_based',
        'stride': 3,
        'batch_size': 3
    }
    
    from utils.inference_scheduler import create_inference_scheduler, BatchInferenceProcessor
    stream_optimized.use_inference_scheduler = True
    stream_optimized.inference_scheduler = create_inference_scheduler(stream_optimized.config['stream_config'])
    stream_optimized.batch_processor = BatchInferenceProcessor(stream_optimized.preprocessor, stream_optimized.device)
    
    # 处理相同数量的帧
    for i in range(1000):
        skeleton_data = np.random.rand(25, 3).astype(np.float32)
        stream_optimized.inference_scheduler.feed_frame(skeleton_data, i, time.time())
    
    optimized_memory = process.memory_info().rss / 1024 / 1024  # MB
    optimized_usage = optimized_memory - initial_memory
    
    logger.info(f"优化方法内存使用: {optimized_usage:.1f} MB")
    
    if traditional_usage > 0:
        memory_improvement = (traditional_usage - optimized_usage) / traditional_usage * 100
        logger.info(f"内存使用改善: {memory_improvement:+.1f}%")
    
    return {
        'traditional_memory_mb': traditional_usage,
        'optimized_memory_mb': optimized_usage,
        'memory_improvement_percent': memory_improvement if traditional_usage > 0 else 0
    }


def main():
    """主测试函数"""
    logger.info("开始真实推理场景测试...")
    
    # 测试1: 推理策略对比
    strategy_results = compare_inference_strategies()
    
    # 测试2: 内存效率测试
    memory_results = test_memory_efficiency()
    
    # 输出最终总结
    logger.info("\n" + "="*60)
    logger.info("测试总结")
    logger.info("="*60)
    
    logger.info("推理策略测试结果:")
    for name, result in strategy_results.items():
        if 'error' not in result:
            logger.info(f"  {name}: {result['fps']:.1f} FPS, 推理比率: {result['inference_ratio']:.3f}")
        else:
            logger.info(f"  {name}: 测试失败 - {result['error']}")
    
    logger.info(f"\n内存效率测试结果:")
    logger.info(f"  传统方法: {memory_results['traditional_memory_mb']:.1f} MB")
    logger.info(f"  优化方法: {memory_results['optimized_memory_mb']:.1f} MB")
    logger.info(f"  内存改善: {memory_results['memory_improvement_percent']:+.1f}%")
    
    logger.info("\n推荐配置:")
    logger.info("  对于实时性要求高的场景: stride=3, batch_size=1")
    logger.info("  对于吞吐量要求高的场景: stride=3, batch_size=3")
    logger.info("  对于负载变化大的场景: 使用自适应策略")


if __name__ == "__main__":
    main()
