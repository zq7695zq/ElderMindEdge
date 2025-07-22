#!/usr/bin/env python3
"""
推理调度器演示脚本
展示如何使用新的推理调度器优化视频流推理
"""

import time
import logging
import numpy as np
from utils.inference_scheduler import InferenceScheduler, SamplingStrategy

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """演示基本使用方法"""
    logger.info("="*60)
    logger.info("推理调度器基本使用演示")
    logger.info("="*60)
    
    # 创建推理调度器
    scheduler = InferenceScheduler(
        window_size=64,      # 窗口大小
        stride=3,            # 步幅大小
        batch_size=3,        # 批量大小
        strategy=SamplingStrategy.STRIDE_BASED,  # 采样策略
        max_buffer_size=200  # 最大缓冲区大小
    )
    
    logger.info(f"创建推理调度器:")
    logger.info(f"  窗口大小: {scheduler.window_size}")
    logger.info(f"  步幅大小: {scheduler.stride}")
    logger.info(f"  批量大小: {scheduler.batch_size}")
    logger.info(f"  采样策略: {scheduler.strategy.value}")
    
    # 模拟视频帧输入
    logger.info(f"\n开始模拟视频帧输入...")
    
    batch_count = 0
    for frame_id in range(200):
        # 生成模拟骨架数据
        skeleton_data = np.random.rand(25, 3).astype(np.float32)
        skeleton_data[:, 2] = np.random.uniform(0.5, 1.0, 25)  # 置信度
        
        # 输入帧到调度器
        batch = scheduler.feed_frame(skeleton_data, frame_id, time.time())
        
        if batch is not None:
            batch_count += 1
            logger.info(f"批次 {batch.batch_id}: 包含 {len(batch.segments)} 个片段")
            
            # 显示片段信息
            for i, segment in enumerate(batch.segments):
                logger.info(f"  片段 {i+1}: 帧 {segment.start_frame_id}-{segment.end_frame_id}")
    
    # 显示最终统计
    stats = scheduler.get_stats()
    logger.info(f"\n处理完成:")
    logger.info(f"  总输入帧数: {stats['total_frames']}")
    logger.info(f"  生成批次数: {stats['processed_batches']}")
    logger.info(f"  生成片段数: {stats['processed_segments']}")
    logger.info(f"  平均批次大小: {stats['avg_batch_size']:.2f}")
    logger.info(f"  缓冲区利用率: {stats['buffer_utilization']:.2%}")


def demo_strategy_comparison():
    """演示不同采样策略的对比"""
    logger.info("\n" + "="*60)
    logger.info("不同采样策略对比演示")
    logger.info("="*60)
    
    strategies = [
        ("滑动窗口 (stride=1)", SamplingStrategy.SLIDING_WINDOW, 1),
        ("步幅采样 (stride=3)", SamplingStrategy.STRIDE_BASED, 3),
        ("步幅采样 (stride=8)", SamplingStrategy.STRIDE_BASED, 8),
        ("自适应采样", SamplingStrategy.ADAPTIVE, 3)
    ]
    
    num_frames = 150
    
    for name, strategy, stride in strategies:
        logger.info(f"\n测试策略: {name}")
        logger.info("-" * 40)
        
        scheduler = InferenceScheduler(
            window_size=64,
            stride=stride,
            batch_size=2,
            strategy=strategy,
            max_buffer_size=100
        )
        
        batch_count = 0
        segment_count = 0
        
        start_time = time.time()
        
        for frame_id in range(num_frames):
            skeleton_data = np.random.rand(25, 3).astype(np.float32)
            batch = scheduler.feed_frame(skeleton_data, frame_id, time.time())
            
            if batch is not None:
                batch_count += 1
                segment_count += len(batch.segments)
        
        processing_time = time.time() - start_time
        stats = scheduler.get_stats()
        
        logger.info(f"  处理时间: {processing_time:.3f}s")
        logger.info(f"  生成批次: {batch_count}")
        logger.info(f"  生成片段: {segment_count}")
        logger.info(f"  片段/帧比率: {segment_count/num_frames:.3f}")
        logger.info(f"  处理速度: {num_frames/processing_time:.1f} FPS")


def demo_batch_inference_simulation():
    """演示批量推理模拟"""
    logger.info("\n" + "="*60)
    logger.info("批量推理模拟演示")
    logger.info("="*60)
    
    # 创建调度器
    scheduler = InferenceScheduler(
        window_size=32,  # 较小的窗口用于演示
        stride=4,
        batch_size=3,
        strategy=SamplingStrategy.STRIDE_BASED
    )
    
    logger.info("模拟批量推理流程:")
    logger.info("1. 输入视频帧")
    logger.info("2. 调度器生成批次")
    logger.info("3. 模拟模型推理")
    logger.info("4. 处理推理结果")
    
    total_inference_time = 0
    inference_count = 0
    
    for frame_id in range(100):
        # 输入帧
        skeleton_data = np.random.rand(25, 3).astype(np.float32)
        batch = scheduler.feed_frame(skeleton_data, frame_id, time.time())
        
        if batch is not None:
            # 模拟批量推理
            inference_start = time.time()
            
            logger.info(f"\n执行批量推理 (批次 {batch.batch_id}):")
            logger.info(f"  包含片段数: {len(batch.segments)}")
            
            # 模拟推理延迟
            simulated_inference_time = 0.01 * len(batch.segments)  # 每个片段10ms
            time.sleep(simulated_inference_time)
            
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            inference_count += 1
            
            logger.info(f"  推理时间: {inference_time:.3f}s")
            logger.info(f"  平均每片段: {inference_time/len(batch.segments):.3f}s")
            
            # 模拟处理结果
            for i, segment in enumerate(batch.segments):
                # 模拟推理结果
                confidence = np.random.uniform(0.6, 0.95)
                action_id = np.random.randint(0, 60)
                
                logger.info(f"    片段 {i+1}: 动作 {action_id}, 置信度 {confidence:.3f}")
    
    # 显示推理统计
    if inference_count > 0:
        avg_inference_time = total_inference_time / inference_count
        logger.info(f"\n推理统计:")
        logger.info(f"  总推理次数: {inference_count}")
        logger.info(f"  总推理时间: {total_inference_time:.3f}s")
        logger.info(f"  平均推理时间: {avg_inference_time:.3f}s")
        logger.info(f"  推理吞吐量: {scheduler.get_stats()['processed_segments']/total_inference_time:.1f} 片段/秒")


def demo_adaptive_strategy():
    """演示自适应策略的动态调整"""
    logger.info("\n" + "="*60)
    logger.info("自适应策略动态调整演示")
    logger.info("="*60)
    
    # 创建自适应调度器
    adaptive_config = {
        'min_stride': 1,
        'max_stride': 8,
        'load_threshold_high': 0.7,
        'load_threshold_low': 0.3,
        'adjustment_factor': 1.5
    }
    
    scheduler = InferenceScheduler(
        window_size=64,
        stride=3,
        batch_size=2,
        strategy=SamplingStrategy.ADAPTIVE,
        max_buffer_size=50,  # 较小的缓冲区以触发自适应调整
        adaptive_config=adaptive_config
    )
    
    logger.info("自适应配置:")
    logger.info(f"  最小步幅: {adaptive_config['min_stride']}")
    logger.info(f"  最大步幅: {adaptive_config['max_stride']}")
    logger.info(f"  高负载阈值: {adaptive_config['load_threshold_high']:.1%}")
    logger.info(f"  低负载阈值: {adaptive_config['load_threshold_low']:.1%}")
    
    logger.info(f"\n开始自适应调整演示:")
    
    # 模拟不同负载情况
    phases = [
        ("低负载阶段", 0.5, 50),    # 50%的帧输入速率，50帧
        ("高负载阶段", 1.5, 50),    # 150%的帧输入速率，50帧
        ("正常负载阶段", 1.0, 50)   # 100%的帧输入速率，50帧
    ]
    
    frame_id = 0
    
    for phase_name, load_factor, num_frames in phases:
        logger.info(f"\n{phase_name} (负载因子: {load_factor}x):")
        logger.info("-" * 30)
        
        for i in range(num_frames):
            skeleton_data = np.random.rand(25, 3).astype(np.float32)
            batch = scheduler.feed_frame(skeleton_data, frame_id, time.time())
            
            # 根据负载因子调整处理速度
            if load_factor < 1.0:
                time.sleep(0.001 * (1.0 - load_factor))  # 低负载时稍微延迟
            
            if batch is not None:
                stats = scheduler.get_stats()
                logger.info(f"  帧 {frame_id}: 批次 {batch.batch_id}, "
                           f"缓冲区利用率 {stats['buffer_utilization']:.1%}")
            
            frame_id += 1
    
    # 显示最终统计
    final_stats = scheduler.get_stats()
    logger.info(f"\n自适应策略最终统计:")
    logger.info(f"  总处理帧数: {final_stats['total_frames']}")
    logger.info(f"  生成批次数: {final_stats['processed_batches']}")
    logger.info(f"  生成片段数: {final_stats['processed_segments']}")
    logger.info(f"  平均缓冲区利用率: {final_stats['buffer_utilization']:.1%}")


def main():
    """主演示函数"""
    logger.info("推理调度器功能演示")
    logger.info("本演示将展示推理调度器的各种功能和优化策略")
    
    # 演示1: 基本使用
    demo_basic_usage()
    
    # 演示2: 策略对比
    demo_strategy_comparison()
    
    # 演示3: 批量推理模拟
    demo_batch_inference_simulation()
    
    # 演示4: 自适应策略
    demo_adaptive_strategy()
    
    logger.info("\n" + "="*60)
    logger.info("演示完成!")
    logger.info("="*60)
    logger.info("推理调度器主要优势:")
    logger.info("1. 降低推理频率: 通过步幅采样减少冗余计算")
    logger.info("2. 批量推理: 提高GPU利用率和推理效率")
    logger.info("3. 自适应调整: 根据系统负载动态优化策略")
    logger.info("4. 内存优化: 更高效的缓冲区管理")
    logger.info("5. 灵活配置: 支持多种采样策略和参数调整")


if __name__ == "__main__":
    main()
