"""
评估用推理模式处理器
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

from action_recognition_stream import ActionRecognitionStream, ActionEvent
from utils.llm_api_client import LLMAPIClient

logger = logging.getLogger(__name__)

@dataclass
class InferenceResult:
    """带时间信息的推理结果"""
    action_id: int
    action_name: str
    confidence: float
    latency: float  # 从视频开始到结果的总延迟
    inference_time: float  # 仅推理花费的时间
    frame_timestamp: float  # 触发结果的帧的时间戳
    enhanced: bool = False
    source: str = "unknown"  # "local", "cloud", 或 "hybrid"

class InferenceMode(ABC):
    """推理模式的抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_complete = False

    @abstractmethod
    def setup(self) -> bool:
        """设置推理模式"""
        pass

    @abstractmethod
    def process_video(self, video_path: str) -> List[InferenceResult]:
        """处理视频并返回带时间的推理结果"""
        pass

    @abstractmethod
    def cleanup(self):
        """清理资源"""
        pass

class LocalInferenceMode(InferenceMode):
    """本地推理模式 - 仅使用本地SkateFormer模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stream = None
        self.results = []

    def setup(self) -> bool:
        """设置本地推理"""
        try:
            # 创建禁用LLM推理的修改配置
            eval_config = self.config.copy()
            eval_config['stream_config'] = eval_config['stream_config'].copy()
            eval_config['stream_config']['llm_inference'] = {'enabled': False}

            # 为评估禁用视频录制
            eval_config['stream_config']['video_recording'] = {'enabled': False}

            # 为客观评估禁用推理调度器优化
            eval_config['stream_config']['inference_scheduler'] = {'enabled': False}

            # 为客观评估禁用目标动作增强
            eval_config['stream_config']['target_actions'] = {'enabled': False}

            # 为流创建临时配置文件
            import tempfile
            import yaml

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(eval_config, f)
                temp_config_path = f.name

            self.stream = ActionRecognitionStream(
                config_path=temp_config_path,
                event_callback=self._on_event,
                enable_video_recording=False
            )

            # 清理临时文件
            os.unlink(temp_config_path)

            self.setup_complete = True
            logger.info("本地推理模式设置完成")
            return True

        except Exception as e:
            logger.error(f"设置本地推理模式失败: {e}")
            return False
    
    def _on_event(self, event: ActionEvent):
        """处理来自流的事件"""
        # 计算延迟（从视频开始到检测）
        latency = event.timestamp

        result = InferenceResult(
            action_id=event.action_id,
            action_name=event.action_name,
            confidence=event.confidence,
            latency=latency,
            inference_time=0.0,  # 本地模式下不单独测量
            frame_timestamp=event.timestamp,
            enhanced=event.enhanced,
            source="local"
        )
        self.results.append(result)

    def process_video(self, video_path: str) -> List[InferenceResult]:
        """仅使用本地推理处理视频"""
        if not self.setup_complete:
            logger.error("本地推理模式未设置")
            return []

        self.results = []

        try:
            logger.info(f"使用本地推理处理视频: {video_path}")

            # 开始处理
            if not self.stream.start_stream(video_path):
                logger.error("启动流失败")
                return []

            # 等待处理完成
            start_time = time.time()
            timeout = 300  # 5分钟超时

            while True:
                status = self.stream.get_status()
                if status.value == "stopped":
                    break
                elif status.value == "error":
                    logger.error("流遇到错误")
                    break
                elif time.time() - start_time > timeout:
                    logger.error("处理超时")
                    break

                time.sleep(0.1)

            logger.info(f"本地推理完成。生成了 {len(self.results)} 个结果")
            return self.results.copy()

        except Exception as e:
            logger.error(f"本地推理错误: {e}")
            return []
        finally:
            if self.stream:
                self.stream.stop_stream()

    def cleanup(self):
        """清理本地推理资源"""
        if self.stream:
            self.stream.stop_stream()
            self.stream = None
        self.setup_complete = False

class CloudInferenceMode(InferenceMode):
    """云推理模式 - 将整个视频上传到LLM进行分析"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_client = None

    def setup(self) -> bool:
        """设置云推理"""
        try:
            # 使用云配置初始化LLM客户端
            llm_config = self.config['stream_config'].get('llm_inference', {})
            if not llm_config.get('enabled', False):
                logger.error("配置中未启用LLM推理")
                return False

            self.llm_client = LLMAPIClient(llm_config)

            self.setup_complete = True
            logger.info("云推理模式设置完成")
            return True

        except Exception as e:
            logger.error(f"设置云推理模式失败: {e}")
            return False
    
    def process_video(self, video_path: str) -> List[InferenceResult]:
        """仅使用云推理处理视频"""
        if not self.setup_complete:
            logger.error("云推理模式未设置")
            return []

        try:
            logger.info(f"使用云推理处理视频: {video_path}")

            # 为整个视频创建虚拟事件
            from action_recognition_stream import ActionEvent

            # 对于云推理，我们将整个视频视为一个事件
            dummy_event = ActionEvent(
                timestamp=0.0,
                action_id=-1,  # 初始未知
                action_name="unknown",
                confidence=0.0,
                enhanced=False,
                frame_id=0,
                video_path=video_path
            )

            start_time = time.time()

            # 对整个视频运行LLM推理
            llm_result = self.llm_client.run_inference(
                dummy_event,
                prompt="请分析这个视频中的主要动作，并给出动作类别和置信度。"
            )

            inference_time = time.time() - start_time

            if llm_result and llm_result.success:
                # 解析LLM结果以提取动作信息
                # 这是一个简化的实现 - 实际上，您需要
                # 解析LLM响应以提取结构化的动作数据
                result = InferenceResult(
                    action_id=0,  # 需要从LLM响应中解析
                    action_name="云检测动作",  # 需要从LLM响应中解析
                    confidence=0.8,  # 需要从LLM响应中解析
                    latency=inference_time,  # 对于云模式，延迟 = 推理时间
                    inference_time=inference_time,
                    frame_timestamp=0.0,
                    enhanced=False,
                    source="cloud"
                )

                logger.info(f"云推理在 {inference_time:.2f}秒内完成")
                return [result]
            else:
                logger.error("云推理失败")
                return []

        except Exception as e:
            logger.error(f"云推理错误: {e}")
            return []

    def cleanup(self):
        """清理云推理资源"""
        self.llm_client = None
        self.setup_complete = False

class HybridInferenceMode(InferenceMode):
    """混合推理模式 - 使用现有系统，同时使用本地和云推理"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stream = None
        self.results = []
        self.llm_results = []

    def setup(self) -> bool:
        """设置混合推理"""
        try:
            # 为流创建临时配置文件
            import tempfile
            import yaml

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(self.config, f)
                temp_config_path = f.name

            # 按原样使用现有系统配置
            self.stream = ActionRecognitionStream(
                config_path=temp_config_path,
                event_callback=self._on_event,
                llm_callback=self._on_llm_result
            )

            # 清理临时文件
            os.unlink(temp_config_path)

            self.setup_complete = True
            logger.info("混合推理模式设置完成")
            return True

        except Exception as e:
            logger.error(f"设置混合推理模式失败: {e}")
            return False
    
    def _on_event(self, event: ActionEvent):
        """处理本地推理事件"""
        latency = event.timestamp

        result = InferenceResult(
            action_id=event.action_id,
            action_name=event.action_name,
            confidence=event.confidence,
            latency=latency,
            inference_time=0.0,  # 不单独测量
            frame_timestamp=event.timestamp,
            enhanced=event.enhanced,
            source="local"
        )
        self.results.append(result)

    def _on_llm_result(self, llm_result, original_event):
        """处理LLM推理结果"""
        if llm_result and llm_result.success:
            # 为LLM推理创建结果
            result = InferenceResult(
                action_id=original_event.action_id,  # 保留原始值用于比较
                action_name=f"llm_{original_event.action_name}",
                confidence=0.9,  # LLM结果通常具有高置信度
                latency=original_event.timestamp + llm_result.inference_time,
                inference_time=llm_result.inference_time,
                frame_timestamp=original_event.timestamp,
                enhanced=False,
                source="cloud"
            )
            self.llm_results.append(result)
    
    def process_video(self, video_path: str) -> List[InferenceResult]:
        """使用混合推理处理视频"""
        if not self.setup_complete:
            logger.error("混合推理模式未设置")
            return []

        self.results = []
        self.llm_results = []

        try:
            logger.info(f"使用混合推理处理视频: {video_path}")

            # 开始处理
            if not self.stream.start_stream(video_path):
                logger.error("启动流失败")
                return []

            # 等待处理完成
            start_time = time.time()
            timeout = 600  # 混合模式10分钟超时

            while True:
                status = self.stream.get_status()
                if status.value == "stopped":
                    break
                elif status.value == "error":
                    logger.error("流遇到错误")
                    break
                elif time.time() - start_time > timeout:
                    logger.error("处理超时")
                    break

                time.sleep(0.1)

            # 合并本地和云结果
            all_results = self.results + self.llm_results
            logger.info(f"混合推理完成。本地: {len(self.results)}, 云: {len(self.llm_results)}")
            return all_results

        except Exception as e:
            logger.error(f"混合推理错误: {e}")
            return []
        finally:
            if self.stream:
                self.stream.stop_stream()

    def cleanup(self):
        """清理混合推理资源"""
        if self.stream:
            self.stream.stop_stream()
            self.stream = None
        self.setup_complete = False

def create_inference_mode(mode_type: str, config: Dict[str, Any]) -> InferenceMode:
    """工厂函数，创建适当的推理模式"""
    if mode_type == 'local':
        return LocalInferenceMode(config)
    elif mode_type == 'cloud':
        return CloudInferenceMode(config)
    elif mode_type == 'hybrid':
        return HybridInferenceMode(config)
    else:
        raise ValueError(f"不支持的推理模式: {mode_type}")
