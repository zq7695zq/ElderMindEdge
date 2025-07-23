#!/usr/bin/env python3
"""
测试新的LLM API客户端
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

def test_llm_api_client():
    """测试LLM API客户端"""
    logger.info("开始测试LLM API客户端")
    
    try:
        # 加载配置
        config = load_config("configs/stream_config.yaml", get_default_config())
        llm_config = config['stream_config'].get('llm_inference', {})
        
        # 创建LLM客户端
        llm_client = LLMAPIClient(llm_config)
        
        # 显示配置信息
        status = llm_client.get_status()
        logger.info(f"LLM客户端状态: {status}")

        # 显示视频上传器状态
        if hasattr(llm_client, 'video_uploader'):
            uploader_status = llm_client.video_uploader.get_status()
            logger.info(f"视频上传器状态: {uploader_status}")

        # 创建模拟事件
        mock_event = MockActionEvent(
            timestamp=time.time(),
            action_id=42,
            action_name="falling",
            confidence=0.65,
            frame_id=1000,
            video_path="test_video.mp4"  # 这是一个模拟路径
        )
        
        # 测试触发条件判断
        should_trigger = llm_client.should_trigger_inference(mock_event)
        logger.info(f"是否应该触发推理: {should_trigger}")
        
        if should_trigger:
            logger.info("根据配置，此事件应该触发LLM推理")
            logger.info("注意: 由于没有真实的视频文件，视频上传会失败，但可以测试流程")

            # 如果你想测试实际的API调用，可以取消下面的注释
            # 注意：需要真实的视频文件才能成功上传到OSS
            # result = llm_client.run_inference(mock_event)
            # if result:
            #     logger.info(f"推理结果: success={result.success}, error={result.error}")
        else:
            logger.info("根据配置，此事件不会触发LLM推理")
        
        # 测试不同置信度的事件
        logger.info("\n测试不同置信度的事件:")
        test_confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for conf in test_confidences:
            test_event = MockActionEvent(
                timestamp=time.time(),
                action_id=8,
                action_name="standing up",
                confidence=conf,
                frame_id=2000,
                video_path="test_video.mp4"
            )
            
            should_trigger = llm_client.should_trigger_inference(test_event)
            logger.info(f"置信度 {conf:.1f}: {'会' if should_trigger else '不会'}触发推理")
        
        # 清理
        llm_client.cleanup()
        logger.info("测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)

if __name__ == "__main__":
    test_llm_api_client()
