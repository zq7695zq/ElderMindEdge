"""
自定义方式LLM推理 - 基于本地GLM-4V模型实现视频理解推理
"""
import os
import time
import torch
import logging
from typing import Dict, Any, Optional
from .llm_inference import BaseLLMInference, LLMInferenceResult, LLMInferenceError

logger = logging.getLogger(__name__)

class CustomLLMInference(BaseLLMInference):
    """基于本地模型的LLM推理实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.processor = None
        
    def initialize(self) -> bool:
        """初始化本地模型"""
        try:
            # 导入必要的库
            try:
                from modelscope import AutoProcessor, Glm4vForConditionalGeneration
            except ImportError:
                raise LLMInferenceError("modelscope library not installed. Please install with: pip install modelscope")
            
            # 获取模型配置
            custom_config = self.config.get('custom_config', {})
            model_path = custom_config.get('model_path')
            
            if not model_path or not os.path.exists(model_path):
                raise LLMInferenceError(f"Model path not found: {model_path}")
            
            device = custom_config.get('device', 'auto')
            torch_dtype = custom_config.get('torch_dtype', 'bfloat16')
            
            logger.info(f"Loading GLM-4V model from: {model_path}")
            
            # 设置torch数据类型
            if hasattr(torch, torch_dtype):
                torch_dtype = getattr(torch, torch_dtype)
            else:
                logger.warning(f"Unknown torch dtype: {torch_dtype}, using bfloat16")
                torch_dtype = torch.bfloat16
            
            # 加载模型
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                device_map=device,
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                use_fast=True
            )
            
            self.is_initialized = True
            logger.info("Custom GLM-4V model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize custom LLM inference: {e}")
            return False
    
    def inference(self, video_path: str, prompt: Optional[str] = None) -> LLMInferenceResult:
        """执行本地模型推理"""
        if not self.is_initialized:
            return LLMInferenceResult(
                success=False,
                error="LLM inference not initialized",
                video_path=video_path,
                timestamp=time.time()
            )
        
        if not os.path.exists(video_path):
            return LLMInferenceResult(
                success=False,
                error=f"Video file not found: {video_path}",
                video_path=video_path,
                timestamp=time.time()
            )
        
        start_time = time.time()
        
        try:
            # 准备提示词
            if not prompt:
                prompt = self._create_default_prompt()
            
            # 创建消息
            messages = self._create_messages(video_path, prompt)
            
            logger.info(f"Starting inference for video: {video_path}")
            
            # 处理输入
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # 获取生成配置
            custom_config = self.config.get('custom_config', {})
            max_new_tokens = custom_config.get('max_new_tokens', 8192)
            
            # 生成结果
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens
            )
            
            # 解码输出
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=False
            )
            
            # 清理输出文本
            output_text = self._clean_output_text(output_text)
            
            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.2f}s, result: {output_text}")
            
            # 解析JSON响应
            parsed_result = self._parse_json_response(output_text)
            
            return LLMInferenceResult(
                success=True,
                result=parsed_result,
                raw_response=output_text,
                inference_time=inference_time,
                video_path=video_path,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Custom inference error: {e}")
            return LLMInferenceResult(
                success=False,
                error=str(e),
                video_path=video_path,
                timestamp=time.time()
            )
    
    def _create_messages(self, video_path: str, prompt: str) -> list:
        """创建推理消息"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]
        return messages
    
    def _clean_output_text(self, text: str) -> str:
        """清理输出文本"""
        # 移除特殊标记
        text = text.replace('<|endoftext|>', '').strip()
        
        # 移除可能的前缀
        if text.startswith('assistant\n'):
            text = text[len('assistant\n'):].strip()
        
        return text
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("Custom LLM inference cleaned up")
