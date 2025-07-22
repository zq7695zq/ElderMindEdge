"""
API方式LLM推理 - 基于智谱AI API实现视频理解推理
"""
import os
import time
import logging
from typing import Dict, Any, Optional
from .llm_inference import BaseLLMInference, LLMInferenceResult, LLMInferenceError
from ..video_uploader import VideoUploader

logger = logging.getLogger(__name__)

class APILLMInference(BaseLLMInference):
    """基于API的LLM推理实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.video_uploader = None
        
    def initialize(self) -> bool:
        """初始化API客户端"""
        try:
            # 导入智谱AI库
            try:
                from zhipuai import ZhipuAI
            except ImportError:
                raise LLMInferenceError("zhipuai library not installed. Please install with: pip install zhipuai")
            
            # 获取API配置
            api_config = self.config.get('api_config', {})
            api_key = api_config.get('api_key')
            
            if not api_key or api_key == "YOUR_API_KEY":
                raise LLMInferenceError("API key not configured. Please set api_key in config.")
            
            # 初始化客户端
            base_url = api_config.get('base_url')
            if base_url:
                self.client = ZhipuAI(api_key=api_key, base_url=base_url)
            else:
                self.client = ZhipuAI(api_key=api_key)
            
            # 初始化视频上传器 - API模式必须要有视频上传器
            upload_config = self.config.get('video_upload', {})
            if upload_config.get('enabled', True):
                self.video_uploader = VideoUploader(upload_config)
                if not self.video_uploader.initialize():
                    logger.error("Video uploader initialization failed, API mode requires video upload")
                    raise LLMInferenceError("API mode requires video upload to be configured")
            else:
                logger.error("Video upload is disabled, but API mode requires video upload")
                raise LLMInferenceError("API mode requires video upload to be enabled")
            
            self.is_initialized = True
            logger.info("API LLM inference initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize API LLM inference: {e}")
            return False
    
    def inference(self, video_path: str, prompt: Optional[str] = None) -> LLMInferenceResult:
        """执行API推理"""
        if not self.is_initialized:
            return LLMInferenceResult(
                success=False,
                error="LLM inference not initialized",
                video_path=video_path,
                timestamp=time.time()
            )
        
        start_time = time.time()
        
        try:
            # 准备视频URL
            video_url = self._prepare_video_url(video_path)
            if not video_url:
                return LLMInferenceResult(
                    success=False,
                    error="Failed to prepare video URL",
                    video_path=video_path,
                    timestamp=time.time()
                )
            
            # 准备提示词
            if not prompt:
                prompt = self._create_default_prompt()
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": video_url
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # 获取API配置
            api_config = self.config.get('api_config', {})
            model = api_config.get('model', 'glm-4.1v-thinking-flashx')
            timeout = api_config.get('timeout', 60)
            max_retries = api_config.get('max_retries', 3)
            
            # 执行推理（带重试）
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=timeout
                    )
                    
                    # 提取响应内容
                    raw_response = response.choices[0].message.content
                    
                    # 解析JSON响应
                    parsed_result = self._parse_json_response(raw_response)
                    
                    inference_time = time.time() - start_time
                    
                    return LLMInferenceResult(
                        success=True,
                        result=parsed_result,
                        raw_response=raw_response,
                        inference_time=inference_time,
                        video_path=video_path,
                        timestamp=time.time()
                    )
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"API inference attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(1)
                    else:
                        logger.error(f"API inference failed after {max_retries} attempts: {e}")
            
            return LLMInferenceResult(
                success=False,
                error=f"API inference failed: {last_error}",
                video_path=video_path,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"API inference error: {e}")
            return LLMInferenceResult(
                success=False,
                error=str(e),
                video_path=video_path,
                timestamp=time.time()
            )
    
    def _prepare_video_url(self, video_path: str) -> Optional[str]:
        """准备视频URL - API模式必须上传到云存储"""
        try:
            # 如果已经是URL，直接返回
            if video_path.startswith(('http://', 'https://')):
                logger.info(f"Using existing URL: {video_path}")
                return video_path

            # 检查本地文件是否存在
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None

            # API模式必须上传到云存储
            if not self.video_uploader:
                logger.error("Video uploader not available, cannot proceed with API inference")
                return None

            logger.info(f"Uploading video to cloud storage: {video_path}")
            upload_result = self.video_uploader.upload_video(video_path)

            if upload_result.get('success'):
                video_url = upload_result.get('url')
                logger.info(f"Video uploaded successfully: {video_url}")
                return video_url
            else:
                error_msg = upload_result.get('error', 'Unknown upload error')
                logger.error(f"Video upload failed: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Failed to prepare video URL: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        if self.video_uploader:
            self.video_uploader.cleanup()
        self.client = None
        self.is_initialized = False
        logger.info("API LLM inference cleaned up")
