"""
LLM推理管理器 - 统一管理API和自定义方式的LLM推理
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional, Callable
from .llm_inference import (
    BaseLLMInference, LLMInferenceResult, LLMInferenceMode,
    create_llm_inference, LLMInferenceError
)
from .llm_rate_limiter import create_rate_limiter, RateLimitExceeded

logger = logging.getLogger(__name__)

class LLMInferenceManager:
    """LLM推理管理器"""
    
    def __init__(self, config: Dict[str, Any], event_callback: Optional[Callable] = None):
        self.config = config
        self.event_callback = event_callback
        self.inference_engine = None
        self.current_mode = None
        self.is_enabled = config.get('enabled', False)
        self.results_dir = None

        # 初始化限流器
        rate_limiter_config = config.get('rate_limiter', {})
        self.rate_limiter = create_rate_limiter(rate_limiter_config)

        # 初始化结果保存目录
        result_config = config.get('result_processing', {})
        if result_config.get('save_results', False):
            self.results_dir = result_config.get('results_dir', 'llm_results')
            os.makedirs(self.results_dir, exist_ok=True)
    
    def initialize(self) -> bool:
        """初始化LLM推理管理器"""
        if not self.is_enabled:
            logger.info("LLM inference is disabled")
            return True
        
        try:
            # 获取推理模式
            mode_str = self.config.get('mode', 'api').lower()
            if mode_str == 'api':
                self.current_mode = LLMInferenceMode.API
            elif mode_str == 'custom':
                self.current_mode = LLMInferenceMode.CUSTOM
            else:
                raise LLMInferenceError(f"Unsupported inference mode: {mode_str}")
            
            # 创建推理引擎
            self.inference_engine = create_llm_inference(self.current_mode, self.config)
            
            # 初始化推理引擎
            if not self.inference_engine.initialize():
                raise LLMInferenceError("Failed to initialize inference engine")
            
            logger.info(f"LLM inference manager initialized with mode: {mode_str}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM inference manager: {e}")
            return False
    
    def should_trigger_inference(self, event) -> bool:
        """判断是否应该触发LLM推理"""
        if not self.is_enabled or not self.inference_engine:
            return False
        
        # print(f'事件触发, 置信度: {event.confidence}')

        # 检查是否有视频路径
        if not event.video_path or not os.path.exists(event.video_path):
            logger.warning(f"Video path not available for LLM inference: {event.video_path}")
            return False

        trigger_config = self.config.get('trigger_conditions', {})

        # 检查是否对所有事件进行推理
        if trigger_config.get('all_events', False):
            return True

        # 检查置信度阈值 - 如果DNN置信度太高，说明很确定，不需要LLM推理
        max_confidence = trigger_config.get('max_confidence', 0.8)
        if event.confidence > max_confidence:
            logger.debug(f"Event confidence {event.confidence} above threshold {max_confidence}, DNN is confident enough")
            return False

        # 检查最小置信度阈值 - 如果置信度太低，也不值得推理
        min_confidence = trigger_config.get('min_confidence', 0.3)
        if event.confidence < min_confidence:
            logger.debug(f"Event confidence {event.confidence} below minimum threshold {min_confidence}")
            return False

        # 检查是否仅对目标动作进行推理
        if trigger_config.get('target_actions_only', True):
            if not event.enhanced:  # enhanced表示是目标动作
                logger.debug(f"Event {event.action_name} is not a target action")
                return False

        # 检查是否仅对关键动作进行推理
        if trigger_config.get('critical_actions_only', False):
            # 这里可以根据具体需求定义关键动作
            critical_actions = [42, 8, 0, 7]  # 摔倒、站起、喝水、坐下
            if event.action_id not in critical_actions:
                logger.debug(f"Event {event.action_name} is not a critical action")
                return False

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

            # 使用限流器控制并发和频率
            with self.rate_limiter.acquire():
                logger.debug(f"获取LLM推理许可成功，开始推理: {event.action_name}")

                # 执行推理
                result = self.inference_engine.inference(event.video_path, prompt)

                # 添加事件信息到结果中
                if result.success and result.result:
                    result.result['original_event'] = {
                        'action_id': event.action_id,
                        'action_name': event.action_name,
                        'confidence': event.confidence,
                        'timestamp': event.timestamp,
                        'frame_id': event.frame_id
                    }

                # 保存结果
                if self.results_dir:
                    self._save_result(result, event)

                # 调用回调函数
                if self.event_callback:
                    self.event_callback(result, event)

                return result

        except RateLimitExceeded as e:
            logger.warning(f"LLM推理限流: {e}")
            return LLMInferenceResult(
                success=False,
                error=f"Rate limit exceeded: {e}",
                video_path=event.video_path,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            return LLMInferenceResult(
                success=False,
                error=str(e),
                video_path=event.video_path,
                timestamp=time.time()
            )

    def _save_result(self, result: LLMInferenceResult, event):
        """保存推理结果"""
        try:
            timestamp = int(time.time())
            filename = f"llm_result_{timestamp}_{event.frame_id}.json"
            filepath = os.path.join(self.results_dir, filename)

            # 准备保存的数据
            save_data = {
                'timestamp': result.timestamp,
                'success': result.success,
                'inference_time': result.inference_time,
                'mode': self.current_mode.value,
                'event_info': {
                    'action_id': event.action_id,
                    'action_name': event.action_name,
                    'confidence': event.confidence,
                    'frame_id': event.frame_id,
                    'timestamp': event.timestamp
                }
            }

            # 添加结果数据
            if result.success:
                save_data['result'] = result.result
                save_data['raw_response'] = result.raw_response
            else:
                save_data['error'] = result.error

            # 根据配置决定是否包含视频路径
            result_config = self.config.get('result_processing', {})
            if result_config.get('include_video_path', True):
                save_data['video_path'] = result.video_path

            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"LLM result saved to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save LLM result: {e}")

    def switch_mode(self, new_mode: str) -> bool:
        """切换推理模式"""
        try:
            # 清理当前引擎
            if self.inference_engine:
                self.inference_engine.cleanup()

            # 更新配置
            self.config['mode'] = new_mode

            # 重新初始化
            return self.initialize()

        except Exception as e:
            logger.error(f"Failed to switch LLM inference mode: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        status = {
            'enabled': self.is_enabled,
            'mode': self.current_mode.value if self.current_mode else None,
            'initialized': self.inference_engine is not None and self.inference_engine.is_initialized,
            'results_dir': self.results_dir,
            'rate_limiter': self.rate_limiter.get_status()
        }
        return status

    def cleanup(self):
        """清理资源"""
        if self.inference_engine:
            self.inference_engine.cleanup()
            self.inference_engine = None

        self.current_mode = None
        logger.info("LLM inference manager cleaned up")
