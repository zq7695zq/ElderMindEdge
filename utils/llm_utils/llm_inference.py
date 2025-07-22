"""
LLM推理模块 - 支持API和自定义方式的视频理解推理
"""
import os
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LLMInferenceMode(Enum):
    """LLM推理模式"""
    API = "api"
    CUSTOM = "custom"

@dataclass
class LLMInferenceResult:
    """LLM推理结果数据结构"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    inference_time: Optional[float] = None
    video_path: Optional[str] = None
    timestamp: Optional[float] = None

class BaseLLMInference(ABC):
    """LLM推理基础抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """初始化推理引擎"""
        pass
    
    @abstractmethod
    def inference(self, video_path: str, prompt: Optional[str] = None) -> LLMInferenceResult:
        """执行推理"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """清理资源"""
        pass
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析JSON格式的响应"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # 尝试提取JSON部分
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Failed to parse JSON response: {response}")
        return None
    
    def _create_default_prompt(self) -> str:
        """创建默认提示词"""
        prompt_config = self.config.get('prompt_config', {})
        system_prompt = prompt_config.get('system_prompt', '')
        user_prompt = prompt_config.get('user_prompt', '请仔细分析这个视频中的人体动作行为')
        
        if system_prompt:
            return f"{system_prompt}\n\n{user_prompt}"
        return user_prompt

class LLMInferenceError(Exception):
    """LLM推理异常"""
    pass

def create_llm_inference(mode: LLMInferenceMode, config: Dict[str, Any]) -> BaseLLMInference:
    """工厂方法创建LLM推理实例"""
    if mode == LLMInferenceMode.API:
        from .llm_api_inference import APILLMInference
        return APILLMInference(config)
    elif mode == LLMInferenceMode.CUSTOM:
        from .llm_custom_inference import CustomLLMInference
        return CustomLLMInference(config)
    else:
        raise ValueError(f"Unsupported LLM inference mode: {mode}")
