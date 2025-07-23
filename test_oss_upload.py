#!/usr/bin/env python3
"""
测试OSS上传和LLM API调用的完整流程
"""
import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.llm_api_client import LLMAPIClient
from utils.utils import load_config, get_default_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MockActionEvent:
    """模拟动作事件"""
    timestamp: float
    action_id: int
    action_name: str
    confidence: float
    enhanced: bool = False
    frame_id: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
    video_path: Optional[str] = None

def create_test_video(filename="test_video.mp4"):
    """创建一个简单的测试视频文件"""
    try:
        import cv2
        import numpy as np
        
        # 创建一个简单的测试视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        
        for i in range(60):  # 3秒的视频，20fps
            # 创建一个简单的彩色帧
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = [i*4 % 255, (i*2) % 255, (i*6) % 255]  # 变化的颜色
            
            # 添加一些文字
            cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, 'Test Video', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        logger.info(f"Created test video: {filename}")
        return True
        
    except ImportError:
        logger.warning("OpenCV not available, creating dummy file")
        # 创建一个虚拟文件
        with open(filename, 'wb') as f:
            f.write(b'dummy video content for testing')
        logger.info(f"Created dummy test file: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to create test video: {e}")
        return False

def test_complete_workflow():
    """测试完整的工作流程"""
    logger.info("开始测试完整的LLM推理工作流程")
    
    try:
        # 创建测试视频
        test_video_path = "test_video_for_oss.mp4"
        if not create_test_video(test_video_path):
            logger.error("无法创建测试视频")
            return
        
        # 加载配置
        config = load_config("configs/stream_config.yaml", get_default_config())
        llm_config = config['stream_config'].get('llm_inference', {})
        
        # 创建LLM客户端
        llm_client = LLMAPIClient(llm_config)
        
        # 显示状态
        status = llm_client.get_status()
        logger.info(f"LLM客户端状态: {status}")
        
        if hasattr(llm_client, 'video_uploader'):
            uploader_status = llm_client.video_uploader.get_status()
            logger.info(f"视频上传器状态: {uploader_status}")
        
        # 创建模拟事件（使用真实的视频文件）
        mock_event = MockActionEvent(
            timestamp=time.time(),
            action_id=42,
            action_name="falling",
            confidence=0.65,
            frame_id=1000,
            video_path=test_video_path
        )
        
        logger.info(f"原始事件视频路径: {mock_event.video_path}")
        
        # 测试触发条件判断
        should_trigger = llm_client.should_trigger_inference(mock_event)
        logger.info(f"是否应该触发推理: {should_trigger}")
        
        if should_trigger:
            logger.info("开始执行LLM推理...")
            
            # 执行推理（这会上传视频到OSS）
            result = llm_client.run_inference(mock_event)
            
            if result:
                logger.info(f"推理完成:")
                logger.info(f"  成功: {result.success}")
                logger.info(f"  错误: {result.error}")
                logger.info(f"  推理时间: {result.inference_time:.2f}s")
                logger.info(f"  视频路径: {result.video_path}")
                
                # 检查事件对象的video_path是否已更新
                logger.info(f"更新后的事件视频路径: {mock_event.video_path}")
                
                if mock_event.video_path.startswith('https://'):
                    logger.info("✅ 事件视频路径已成功更新为OSS URL")
                else:
                    logger.warning("❌ 事件视频路径未更新为OSS URL")
            else:
                logger.error("推理返回None")
        else:
            logger.info("根据配置，此事件不会触发LLM推理")
        
        # 清理
        llm_client.cleanup()
        
        # 删除测试文件
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
            logger.info(f"已删除测试文件: {test_video_path}")
        
        logger.info("测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)

if __name__ == "__main__":
    test_complete_workflow()
