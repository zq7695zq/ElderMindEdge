"""
LLM API客户端 - 调用/api/llm/inference接口进行LLM推理
"""
import os
import json
import time
import logging
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .video_uploader import VideoUploader

logger = logging.getLogger(__name__)

@dataclass
class LLMInferenceResult:
    """LLM推理结果"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    inference_time: float = 0.0
    video_path: Optional[str] = None
    timestamp: float = 0.0
    raw_response: Optional[str] = None

class LLMAPIClient:
    """LLM API客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_enabled = config.get('enabled', False)
        self.api_url = config.get('api_url', 'http://localhost:8080/api/llm/inference')
        self.timeout = config.get('timeout', 60)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)

        # 触发条件配置
        self.trigger_conditions = config.get('trigger_conditions', {})

        # 初始化视频上传器
        video_upload_config = config.get('video_upload', {})
        self.video_uploader = VideoUploader(video_upload_config)
        if self.is_enabled:
            if not self.video_uploader.initialize():
                logger.warning("视频上传器初始化失败，LLM推理可能无法正常工作")

        logger.info(f"LLM API客户端初始化: enabled={self.is_enabled}, url={self.api_url}")
    
    def should_trigger_inference(self, event) -> bool:
        """判断是否应该触发LLM推理"""
        if not self.is_enabled:
            return False
        
        # 检查是否有视频路径
        if not event.video_path or not os.path.exists(event.video_path):
            logger.warning(f"Video path not available for LLM inference: {event.video_path}")
            return False

        # 检查是否对所有事件进行推理
        if self.trigger_conditions.get('all_events', False):
            return True

        # 检查置信度阈值 - 如果DNN置信度太高，说明很确定，不需要LLM推理
        max_confidence = self.trigger_conditions.get('max_confidence', 0.8)
        if event.confidence > max_confidence:
            logger.debug(f"Event confidence {event.confidence} above threshold {max_confidence}, DNN is confident enough")
            return False

        # 检查最小置信度阈值
        min_confidence = self.trigger_conditions.get('min_confidence', 0.0)
        if event.confidence < min_confidence:
            logger.debug(f"Event confidence {event.confidence} below threshold {min_confidence}")
            return False

        # 检查是否仅对目标动作进行推理
        if self.trigger_conditions.get('target_actions_only', False):
            # 这里需要根据实际的目标动作配置来判断
            # 暂时返回True，具体逻辑可以后续完善
            pass

        # 检查是否仅对关键动作进行推理
        if self.trigger_conditions.get('critical_actions_only', False):
            # 这里需要根据实际的关键动作配置来判断
            # 暂时返回True，具体逻辑可以后续完善
            pass

        return True
    
    def run_inference(self, event, prompt: Optional[str] = None) -> Optional[LLMInferenceResult]:
        """执行LLM推理"""
        if not self.should_trigger_inference(event):
            return None

        if not event.video_path or not os.path.exists(event.video_path):
            logger.warning(f"Video path not available for LLM inference: {event.video_path}")
            return None

        try:
            logger.info(f"Running LLM inference for event: {event.action_name} (confidence: {event.confidence:.3f})")

            start_time = time.time()

            # 上传视频到OSS并获取URL
            video_url = self.video_uploader.upload_video(event.video_path)
            if not video_url:
                logger.error(f"Failed to upload video to OSS: {event.video_path}")
                return LLMInferenceResult(
                    success=False,
                    error="Failed to upload video to OSS",
                    video_path=event.video_path,
                    timestamp=time.time()
                )

            logger.info(f"Video uploaded to OSS: {video_url}")

            # 更新event对象中的video_path为OSS URL
            original_video_path = event.video_path
            event.video_path = video_url
            logger.info(f"Updated event video_path from {original_video_path} to {video_url}")

            # 准备请求数据，匹配Spring Boot接口格式
            request_data = {
                'videoUrl': video_url,  # 使用驼峰命名匹配Java接口
                'customPrompt': prompt if prompt else f"请分析这个视频中的动作行为。检测到的动作：{event.action_name}，置信度：{event.confidence:.2f}"
            }

            # 可以在日志中记录事件信息，但不发送给API（因为接口不需要）
            logger.info(f"Event info - ID: {event.action_id}, Name: {event.action_name}, "
                       f"Confidence: {event.confidence:.3f}, Frame: {event.frame_id}")
            
            # 执行API调用，带重试机制
            result = self._call_api_with_retry(request_data)
            
            inference_time = time.time() - start_time
            
            if result:
                logger.info(f"LLM inference completed in {inference_time:.2f}s for event: {event.action_name}")
                return LLMInferenceResult(
                    success=True,
                    result=result,
                    inference_time=inference_time,
                    video_path=event.video_path,
                    timestamp=time.time()
                )
            else:
                return LLMInferenceResult(
                    success=False,
                    error="API call failed",
                    inference_time=inference_time,
                    video_path=event.video_path,
                    timestamp=time.time()
                )
                
        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return LLMInferenceResult(
                success=False,
                error=str(e),
                video_path=event.video_path,
                timestamp=time.time()
            )
    
    def _call_api_with_retry(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """带重试机制的API调用"""
        last_error = None
        print(request_data)
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    self.api_url,
                    json=request_data,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"API call successful: {result}")
                    return result
                else:
                    error_msg = f"API call failed with status {response.status_code}: {response.text}"
                    logger.warning(error_msg)
                    last_error = error_msg
                    
            except requests.exceptions.Timeout:
                error_msg = f"API call timeout after {self.timeout}s"
                logger.warning(error_msg)
                last_error = error_msg
                
            except requests.exceptions.ConnectionError:
                error_msg = "API connection error"
                logger.warning(error_msg)
                last_error = error_msg
                
            except Exception as e:
                error_msg = f"API call error: {e}"
                logger.warning(error_msg)
                last_error = error_msg
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries - 1:
                logger.debug(f"Retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)
        
        logger.error(f"API call failed after {self.max_retries} attempts. Last error: {last_error}")
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            'enabled': self.is_enabled,
            'api_url': self.api_url,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'trigger_conditions': self.trigger_conditions
        }
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'video_uploader'):
            self.video_uploader.cleanup()
        logger.info("LLM API client cleaned up")
