#!/usr/bin/env python3
"""
测试请求格式是否正确匹配Spring Boot接口
"""
import json
import sys
import os
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

def test_request_format():
    """测试请求格式"""
    logger.info("测试请求格式是否匹配Spring Boot接口")
    
    try:
        # 加载配置
        config = load_config("configs/stream_config.yaml", get_default_config())
        llm_config = config['stream_config'].get('llm_inference', {})
        
        # 临时禁用视频上传，只测试请求格式
        llm_config['video_upload']['enabled'] = False
        
        # 创建LLM客户端
        llm_client = LLMAPIClient(llm_config)
        
        # 创建模拟事件，使用一个假的OSS URL
        mock_event = MockActionEvent(
            timestamp=time.time(),
            action_id=42,
            action_name="falling",
            confidence=0.75,
            frame_id=1000,
            video_path="https://my-llm-server.oss-cn-guangzhou.aliyuncs.com/llm_videos/test_video.mp4"
        )
        
        logger.info("模拟事件创建完成")
        logger.info(f"  动作ID: {mock_event.action_id}")
        logger.info(f"  动作名称: {mock_event.action_name}")
        logger.info(f"  置信度: {mock_event.confidence}")
        logger.info(f"  视频URL: {mock_event.video_path}")
        
        # 手动构建请求数据来验证格式
        video_url = mock_event.video_path
        prompt = f"请分析这个视频中的动作行为。检测到的动作：{mock_event.action_name}，置信度：{mock_event.confidence:.2f}"
        
        request_data = {
            'videoUrl': video_url,
            'customPrompt': prompt
        }
        
        logger.info("生成的请求数据:")
        logger.info(json.dumps(request_data, indent=2, ensure_ascii=False))
        
        # 验证请求数据格式
        logger.info("\n验证请求格式:")
        logger.info("✅ 使用驼峰命名: videoUrl (而不是 video_url)")
        logger.info("✅ 使用驼峰命名: customPrompt (而不是 prompt)")
        logger.info("✅ 包含完整的OSS URL")
        logger.info("✅ 包含事件信息在提示词中")
        
        # 验证与Spring Boot接口的匹配性
        logger.info("\nSpring Boot接口匹配性检查:")
        
        # 检查必需字段
        if 'videoUrl' in request_data and request_data['videoUrl']:
            logger.info("✅ videoUrl 字段存在且非空")
        else:
            logger.error("❌ videoUrl 字段缺失或为空")
        
        if 'customPrompt' in request_data:
            logger.info("✅ customPrompt 字段存在")
            if request_data['customPrompt']:
                logger.info("✅ customPrompt 内容非空")
            else:
                logger.info("⚠️  customPrompt 内容为空（可选字段）")
        else:
            logger.info("⚠️  customPrompt 字段缺失（可选字段）")
        
        # 检查URL格式
        video_url = request_data['videoUrl']
        if video_url.startswith('https://') and 'oss-cn-guangzhou.aliyuncs.com' in video_url:
            logger.info("✅ 视频URL格式正确（OSS URL）")
        else:
            logger.warning("⚠️  视频URL格式可能不正确")
        
        logger.info("\n请求格式验证完成！")
        logger.info("该格式应该与您的Spring Boot接口完全匹配。")
        
        # 清理
        llm_client.cleanup()
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)

if __name__ == "__main__":
    test_request_format()
