import cv2
import numpy as np
import os
import time
import threading
import queue
import torch
import torch.nn.functional as F
import yaml
import sys
import pickle
import traceback
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Callable, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
import math
from utils.preprocessor import SkateFormerPreprocessor
from utils.video_recorder import FrameBasedVideoRecorder, FrameData
from utils.yolo2ntu_converter import YOLOToNTUConverter
from utils.utils import (
    import_class, load_config, get_default_config, create_metadata_file, load_action_labels
)
from utils.llm_utils import LLMInferenceManager

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ActionEvent:
    """动作识别事件的数据结构"""
    timestamp: float
    action_id: int
    action_name: str
    confidence: float
    enhanced: bool = False
    frame_id: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    video_path: Optional[str] = None  # 保存的视频片段路径

class StreamStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"

class FramePacket(NamedTuple):
    """帧数据包，用于线程间传递"""
    frame: np.ndarray  # 原始视频帧
    frame_id: int      # 帧ID
    timestamp: float   # 时间戳

class SkeletonPacket(NamedTuple):
    """骨架数据包，用于线程间传递"""
    skeleton: np.ndarray  # 骨架数据
    frame: np.ndarray     # 原始视频帧
    frame_id: int         # 帧ID
    timestamp: float      # 时间戳
    valid_points: int     # 有效关键点数量

class InferenceResult(NamedTuple):
    """推理结果数据包"""
    action_id: int        # 动作ID
    action_name: str      # 动作名称
    confidence: float     # 置信度
    enhanced: bool        # 是否增强
    frame_id: int         # 帧ID
    timestamp: float      # 时间戳
    frame: np.ndarray     # 原始视频帧

class ActionRecognitionStream:
    """主动作识别流处理器"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 yolo_model_path: Optional[str] = None,
                 skateformer_config_path: Optional[str] = None,
                 skateformer_weights_path: Optional[str] = None,
                 target_actions: Optional[List[int]] = None,
                 boost_factor: Optional[float] = None,
                 fps_target: Optional[int] = None,
                 event_callback: Optional[Callable[[ActionEvent], None]] = None,
                 enable_video_recording: bool = True,
                 video_output_dir: Optional[str] = None,
                 pre_event_seconds: float = 5.0,
                 post_event_seconds: float = 5.0,
                 llm_callback: Optional[Callable] = None):
        
        self.config = self._load_config(config_path)
        
        if yolo_model_path: self.config['stream_config']['yolo_model_path'] = yolo_model_path
        if skateformer_config_path: self.config['stream_config']['skateformer_config_path'] = skateformer_config_path
        if skateformer_weights_path: self.config['stream_config']['skateformer_weights_path'] = skateformer_weights_path
        if target_actions is not None:
            self.config['stream_config']['target_actions']['actions'] = [{'id': action_id, 'name': f'action_{action_id}', 'priority': 'medium'} for action_id in target_actions]
        if boost_factor is not None: self.config['stream_config']['target_actions']['boost_factor'] = boost_factor
        if fps_target is not None: self.config['stream_config']['fps_target'] = fps_target
        
        self.enable_video_recording = enable_video_recording
        video_config = self.config['stream_config'].get('video_recording', {})
        
        if enable_video_recording:
            fps_target_val = self.config['stream_config'].get('fps_target', 30)
            pre_event_sec = pre_event_seconds if pre_event_seconds is not None else video_config.get('pre_event_seconds', 5.0)
            post_event_sec = post_event_seconds if post_event_seconds is not None else video_config.get('post_event_seconds', 5.0)
            pre_event_frames = int(pre_event_sec * fps_target_val)
            post_event_frames = int(post_event_sec * fps_target_val)

            self.video_recorder = FrameBasedVideoRecorder(
                output_dir=video_output_dir or video_config.get('output_dir', 'event_clips'),
                pre_event_frames=pre_event_frames,
                post_event_frames=post_event_frames,
                fps=fps_target_val,
                max_buffer_size=video_config.get('max_buffer_size', 300),
            )
            logger.info(f"已启用基于帧的视频录制器：事件前 {pre_event_frames} 帧，事件后 {post_event_frames} 帧")
        else:
            self.video_recorder = None
        
        use_gpu = self.config['stream_config'].get('performance', {}).get('use_gpu', True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        self.yolo_converter = YOLOToNTUConverter(self.config['stream_config']['yolo_model_path'])
        self.skateformer_model, self.model_config = self._load_skateformer_model(
            self.config['stream_config']['skateformer_config_path'], 
            self.config['stream_config']['skateformer_weights_path'])
        
        model_args = self.model_config.get('model_args', {})
        self.num_people = model_args.get('num_people', 2)
        self.num_classes = model_args.get('num_classes', 60)
        self.dataset_type = "ntu120" if self.num_classes == 120 else "ntu60"
        
        window_size = self.config['stream_config'].get('window_size', 64)
        self.preprocessor = SkateFormerPreprocessor(window_size=window_size, num_people=self.num_people)
        
        self.priority_boost_factors = self.config['stream_config'].get('priority_boost_factors', {'critical': 5.0, 'high': 3.0, 'medium': 2.0, 'low': 1.5})
        self.base_boost_factor = self.config['stream_config']['target_actions'].get('boost_factor', 3.0)
        self.target_actions = self._parse_target_actions()
        
        self.event_filtering = self.config['stream_config'].get('event_filtering', {
            'enabled': True, 
            'min_confidence': 0.1, 
            'duplicate_suppression': True, 
            'duplicate_frame_window': 30
        })
        
        self.last_event_frame: Dict[int, int] = {}
        self.current_action_id: Optional[int] = None
        
        self.suppress_inference_until_ts: float = 0.0

        self.fps_target = self.config['stream_config'].get('fps_target', 30)
        self.frame_interval = 1.0 / self.fps_target
        
        self.status = StreamStatus.STOPPED
        self.event_callback = event_callback
        self.frame_count = 0

        self.processing_thread = None
        self.stop_event = threading.Event()

        # FIX: 移除多进程池，因为它导致序列化和CUDA错误
        # self.thread_pool = concurrent.futures.ProcessPoolExecutor(max_workers=8)

        # FIX: 添加一个列表来管理工作线程
        self.worker_threads: List[threading.Thread] = []
        
        # FIX: 添加线程锁以保护共享数据
        self.stats_lock = threading.Lock()
        self.event_lock = threading.Lock()

        queue_config = self.config['stream_config'].get('queue_config', {})
        frame_queue_size = queue_config.get('frame_queue_size', 50) 
        skeleton_queue_size = queue_config.get('skeleton_queue_size', 50)
        inference_queue_size = queue_config.get('inference_queue_size', 10)

        self.frame_queue = queue.Queue(maxsize=frame_queue_size)
        self.skeleton_queue = queue.Queue(maxsize=skeleton_queue_size)
        self.inference_queue = queue.Queue(maxsize=inference_queue_size)
        self.video_queue = queue.Queue(maxsize=10)
        self.llm_queue = queue.Queue(maxsize=5)

        self.performance_stats = {
            'frame_capture_fps': 0.0, 'skeleton_detection_fps': 0.0,
            'inference_fps': 0.0, 'dropped_frames': 0, 'queue_sizes': {}
        }

        self.action_labels = self._load_action_labels()

        llm_config = self.config['stream_config'].get('llm_inference', {})
        self.llm_manager = LLMInferenceManager(llm_config, llm_callback)
        if not self.llm_manager.initialize():
            logger.warning("LLM推理管理器初始化失败")

        logger.info("动作识别流初始化成功")
        logger.info(f"模型: {self.num_people}人, {self.num_classes}类, 数据集: {self.dataset_type}")
        logger.info(f"目标动作: {len(self.target_actions)}个动作已启用增强")
        logger.info(f"视频录制: {'启用' if self.enable_video_recording else '禁用'}")
        logger.info(f"LLM推理: {self.llm_manager.get_status()}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        return load_config(config_path or "configs/stream_config.yaml", get_default_config())

    def _get_default_config(self) -> Dict[str, Any]:
        return get_default_config()

    def _load_action_labels(self) -> Dict[int, str]:
        return load_action_labels(self.dataset_type)
    
    def _parse_target_actions(self) -> Dict[int, Dict[str, Any]]:
        target_actions = {}
        target_actions_config = self.config['stream_config'].get('target_actions', {})
        if not target_actions_config.get('enabled', False): return target_actions
        actions_list = target_actions_config.get('actions', [])
        if not actions_list: return target_actions
        for action_config in actions_list:
            action_id = action_config.get('id')
            if action_id is None: continue
            action_name = action_config.get('name', f'action_{action_id}')
            action_priority = action_config.get('priority', 'medium')
            target_actions[action_id] = {
                'name': action_name, 'priority': action_priority,
                'boost_factor': self.priority_boost_factors.get(action_priority, self.base_boost_factor)
            }
        return target_actions
    
    def _load_skateformer_model(self, config_path: str, weights_path: str):
        if not os.path.exists(config_path): raise FileNotFoundError(f"配置文件未找到: {config_path}")
        if not os.path.exists(weights_path): raise FileNotFoundError(f"权重文件未找到: {weights_path}")
        
        with open(config_path, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
        
        Model = import_class(config.get('model', 'model.SkateFormer.SkateFormer'))
        model = Model(**config['model_args'])
        
        weights = torch.load(weights_path, map_location=self.device)
        
        processed_weights = OrderedDict()
        for k, v in weights.items():
            processed_weights[k.split('module.')[-1]] = v
        
        try:
            model.load_state_dict(processed_weights)
        except Exception as e:
            logger.warning(f"权重部分加载: {e}")
            state = model.state_dict()
            state.update(processed_weights)
            model.load_state_dict(state)
        
        model.to(self.device)
        model.eval()
        
        logger.info("SkateFormer模型加载成功")
        return model, config
    
    def start_stream(self, stream_source: str) -> bool:
        if self.status != StreamStatus.STOPPED:
            logger.warning("流已经在运行或正在启动")
            return False

        self.status = StreamStatus.STARTING
        self.stop_event.clear()
        self.frame_count = 0 

        # 重置性能统计数据
        with self.stats_lock:
            self.performance_stats = {
                'frame_capture_fps': 0.0, 'skeleton_detection_fps': 0.0,
                'inference_fps': 0.0, 'dropped_frames': 0, 'queue_sizes': {}
            }
        
        # 重置事件记录
        with self.event_lock:
            self.last_event_frame.clear()

        self.processing_thread = threading.Thread(target=self._process_stream, args=(stream_source,), daemon=True)
        self.processing_thread.start()

        return True

    def _video_recording_worker(self):
        logger.info("视频录制工作线程已启动")
        while not self.stop_event.is_set():
            try:
                # 检查队列是否有任务，超时0.5秒
                video_task = self.video_queue.get(timeout=0.5)
                # 检查是否有None信号以优雅退出
                if video_task is None: break

                task_type = video_task.get('type')

                if task_type == 'start_recording':
                    event, frame = video_task.get('event'), video_task.get('frame')
                    if self.video_recorder and self.enable_video_recording:
                        self.video_recorder.add_frame(frame, event.frame_id)
                        video_path = self.video_recorder.start_recording(event)
                        if video_path:
                            event.video_path = video_path
                            post_event_duration = self.video_recorder.post_event_frames / self.fps_target
                            self.suppress_inference_until_ts = time.time() + post_event_duration
                elif task_type == 'update_recordings':
                    frame, frame_id = video_task.get('frame'), video_task.get('frame_id')
                    if self.video_recorder:
                        for recording_info in self.video_recorder.update_recordings(frame, frame_id):
                            event = recording_info['event']
                            if hasattr(self, 'llm_manager'):
                                try: self.llm_queue.put_nowait({'type': 'llm_inference', 'event': event})
                                except queue.Full: logger.warning("LLM队列已满，丢弃推理任务")
                            if self.event_callback: self.event_callback(event)
            except queue.Empty:
                # 即使队列为空，也检查并处理已完成的录制，以释放资源
                if self.video_recorder and self.video_recorder.has_active_recordings():
                    self.video_recorder.check_finished_recordings()
                continue
            except Exception as e:
                logger.error(f"视频录制错误: {e}", exc_info=False)
        logger.info("视频录制工作线程已停止")
    
    def stop_stream(self):
        if self.status == StreamStatus.STOPPED:
            return
        logger.info("正在停止动作识别流...")
        self.stop_event.set()

        # FIX: 清空队列并放入None信号，以确保阻塞的线程可以退出
        for q in [self.frame_queue, self.skeleton_queue, self.inference_queue, self.video_queue, self.llm_queue]:
            with q.mutex:
                q.queue.clear()
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10.0)
            if self.processing_thread.is_alive():
                logger.warning("主处理线程超时未停止。")

        # FIX: 移除进程池关闭逻辑
        # if hasattr(self, 'thread_pool'):
        #     self.thread_pool.shutdown(wait=False, cancel_futures=True)

        if hasattr(self, 'llm_manager'):
            self.llm_manager.cleanup()

        self.status = StreamStatus.STOPPED
        logger.info("动作识别流已停止")

    def _frame_capture_worker(self, stream_source: str):
        """
        负责从视频流读取帧，写入 frame_queue，
        同时（若启用录制）把每一帧推送给 video_queue，
        以便 _video_recording_worker 调用 update_recordings。
        """
        cap = None
        last_fps_time, fps_counter = time.time(), 0

        try:
            cap = cv2.VideoCapture(stream_source)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开流: {stream_source}")

            # 有些摄像头/文件流支持减少缓冲，避免延迟
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"帧捕获工作线程已启动: {stream_source}")

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.info("视频流已结束或读取帧失败。")
                    break  # 结束循环

                current_time = time.time()
                self.frame_count += 1   # 线程安全：只有此线程写

                # ---------- 1. 送入骨架 & 推理流水线 ----------
                frame_packet = FramePacket(
                    frame=frame,
                    frame_id=self.frame_count,
                    timestamp=current_time
                )
                try:
                    self.frame_queue.put(frame_packet, timeout=1.0)
                except queue.Full:
                    with self.stats_lock:
                        self.performance_stats['dropped_frames'] += 1
                    logger.warning("帧队列已满，丢弃一帧。")
                    # 即便丢弃推理，也继续录制，所以不 return
                # ---------- 2. 送入视频录制流水线 ----------
                if self.enable_video_recording:
                    try:
                        self.video_queue.put_nowait({
                            'type': 'update_recordings',
                            'frame': frame,
                            'frame_id': self.frame_count
                        })
                    except queue.Full:
                        # 丢掉最旧元素再放入，避免录制断帧
                        try:
                            self.video_queue.get_nowait()
                            self.video_queue.put_nowait({
                                'type': 'update_recordings',
                                'frame': frame,
                                'frame_id': self.frame_count
                            })
                        except queue.Empty:
                            pass

                # ---------- 3. FPS 统计 ----------
                fps_counter += 1
                if current_time - last_fps_time >= 1.0:
                    with self.stats_lock:
                        self.performance_stats['frame_capture_fps'] = (
                            fps_counter / (current_time - last_fps_time)
                        )
                    fps_counter, last_fps_time = 0, current_time

                # ---------- 4. 控制采样速率 ----------
                processing_time = time.time() - current_time
                sleep_duration = self.frame_interval - processing_time
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

        except Exception as e:
            logger.error(f"帧捕获错误: {e}", exc_info=True)
            self.status = StreamStatus.ERROR
            self.stop_event.set()
        finally:
            if cap:
                cap.release()
            logger.info("帧捕获工作线程已停止。")
            # 通知下游：不再有新数据
            try:
                self.frame_queue.put_nowait(None)
            except queue.Full:
                pass

    
    def _skeleton_detection_worker(self):
        last_fps_time, fps_counter = time.time(), 0
        skip_frame_count = 0
        max_skip_frames = self.config['stream_config'].get('max_skip_frames', 3)
        scene_change_threshold = self.config['stream_config'].get('scene_change_threshold', 30.0)
        
        last_gray_frame = None
        consecutive_low_activity = 0
        low_activity_threshold = 5
        adaptive_skip_frames = max_skip_frames

        logger.info("骨架检测工作线程已启动")

        while not self.stop_event.is_set():
            try:
                frame_packet = self.frame_queue.get(timeout=1.0)
                if frame_packet is None: break # 收到退出信号

                frame, frame_id, timestamp = frame_packet.frame, frame_packet.frame_id, frame_packet.timestamp

                process_this_frame = True
                current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if skip_frame_count > 0 and last_gray_frame is not None:
                    frame_diff = np.mean(np.abs(current_gray_frame - last_gray_frame))
                    if frame_diff < scene_change_threshold:
                        skip_frame_count -= 1
                        process_this_frame = False
                        consecutive_low_activity += 1
                        if consecutive_low_activity > low_activity_threshold:
                            adaptive_skip_frames = min(max_skip_frames * 2, 10)
                    else:
                        consecutive_low_activity = 0
                        adaptive_skip_frames = max_skip_frames
                        skip_frame_count = 0 

                if process_this_frame:
                    skeleton_data = self.yolo_converter.extract_and_convert(frame)
                    valid_points = np.sum(skeleton_data[:, 2] > 0.3) if skeleton_data is not None and skeleton_data.shape[0] == 25 else 0
                    if valid_points >= self.config['stream_config'].get('min_keypoints', 20):
                        skeleton_packet = SkeletonPacket(
                            skeleton=skeleton_data, frame=frame, frame_id=frame_id,
                            timestamp=timestamp, valid_points=valid_points
                        )
                        try:
                            self.skeleton_queue.put_nowait(skeleton_packet)
                            skip_frame_count = adaptive_skip_frames
                        except queue.Full:
                            # 队列满时，尝试丢弃旧的，放入新的
                            try:
                                self.skeleton_queue.get_nowait()
                                self.skeleton_queue.put_nowait(skeleton_packet)
                            except queue.Empty: pass
                    else:
                        logger.debug(f"跳过帧 {frame_id}, 有效关键点数: {valid_points}")

                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    with self.stats_lock: # FIX: 使用锁保护共享字典
                        self.performance_stats['skeleton_detection_fps'] = fps_counter / (current_time - last_fps_time)
                    fps_counter, last_fps_time = 0, current_time

                last_gray_frame = current_gray_frame

            except queue.Empty: continue
            except Exception as e:
                logger.error(f"骨架检测错误: {e}", exc_info=False)
        
        logger.info("骨架检测工作线程已停止")
        # 传递退出信号
        try:
            self.skeleton_queue.put_nowait(None)
        except queue.Full:
            pass

    def _inference_worker(self):
        last_fps_time, fps_counter, last_inference_time = time.time(), 0, 0
        logger.info("推理工作线程已启动")
        while not self.stop_event.is_set():
            try:
                skeleton_packet = self.skeleton_queue.get(timeout=1.0)
                if skeleton_packet is None: break # 收到退出信号
                if time.time() < self.suppress_inference_until_ts: continue

                preprocessed_data = self.preprocessor.add_frame(skeleton_packet.skeleton)
                current_time = time.time()
                if preprocessed_data is not None and (current_time - last_inference_time) >= self.frame_interval:
                    inference_result = self._run_inference_async(
                        preprocessed_data, skeleton_packet.frame, 
                        skeleton_packet.frame_id, skeleton_packet.timestamp
                    )
                    if inference_result:
                        try: self.inference_queue.put_nowait(inference_result)
                        except queue.Full:
                            try:
                                self.inference_queue.get_nowait()
                                self.inference_queue.put_nowait(inference_result)
                            except queue.Empty: pass
                    last_inference_time = current_time

                fps_counter += 1
                if current_time - last_fps_time >= 1.0:
                    with self.stats_lock: # FIX: 使用锁保护共享字典
                        self.performance_stats['inference_fps'] = fps_counter / (current_time - last_fps_time)
                    fps_counter, last_fps_time = 0, current_time

            except queue.Empty: continue
            except Exception as e: logger.error(f"推理错误: {e}", exc_info=True)
        
        logger.info("推理工作线程已停止")
        try:
            self.inference_queue.put_nowait(None)
        except queue.Full:
            pass

    # FIX: 重构此方法以直接管理线程
    def _process_stream(self, stream_source: str):
        try:
            self.status = StreamStatus.RUNNING
            logger.info(f"流已启动: {stream_source}")

            # 直接创建和管理线程，而不是使用进程池
            self.worker_threads = [
                threading.Thread(target=self._frame_capture_worker, args=(stream_source,)),
                threading.Thread(target=self._skeleton_detection_worker),
                threading.Thread(target=self._inference_worker),
                threading.Thread(target=self._video_recording_worker),
                threading.Thread(target=self._event_processing_worker),
                threading.Thread(target=self._llm_inference_worker),
                threading.Thread(target=self._performance_monitor_worker)
            ]

            for thread in self.worker_threads:
                thread.daemon = True 
                thread.start()

            # 监控线程状态，直到stop_event被设置或有线程崩溃
            while not self.stop_event.is_set():
                if not all(t.is_alive() for t in self.worker_threads):
                    for i, t in enumerate(self.worker_threads):
                        if not t.is_alive():
                            logger.error(f"工作线程 {t.name} (index {i}) 意外终止。正在停止流...")
                    self.status = StreamStatus.ERROR
                    self.stop_event.set() # 发生异常，停止所有线程
                    break
                time.sleep(1.0)

        except Exception as e:
            logger.error(f"流处理主循环错误: {e}", exc_info=True)
            self.status = StreamStatus.ERROR
        finally:
            logger.info("流处理主循环正在关闭...")
            self.stop_event.set()
            
            # 等待所有工作线程执行完毕
            for thread in self.worker_threads:
                if thread.is_alive():
                    thread.join(timeout=5.0)

            if self.video_recorder:
                self.video_recorder.stop_all_recordings()
            
            logger.info("所有工作线程已停止。")
    
    def _run_inference_async(self, preprocessed_data: Tuple[torch.Tensor, torch.Tensor],
                           frame: np.ndarray, frame_id: int, timestamp: float) -> Optional[InferenceResult]:
        try:
            data_tensor, index_tensor = preprocessed_data
            data_tensor = data_tensor.to(self.device, non_blocking=True)
            index_tensor = index_tensor.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                with torch.no_grad():
                    output = self.skateformer_model(data_tensor, index_tensor)

                    enhanced_actions = []
                    target_actions_enabled = self.config['stream_config'].get('target_actions', {}).get('enabled', False)
                    if self.target_actions and target_actions_enabled:
                        _, top5_indices = torch.topk(output, 5, dim=1)
                        for action_id, action_info in self.target_actions.items():
                            if action_id in top5_indices:
                                output[0, action_id] += math.log(action_info['boost_factor'])
                                enhanced_actions.append({'id': action_id, 'name': action_info['name']})
                    
                    probabilities = F.softmax(output, dim=1)

            confidence, predicted_class_tensor = torch.max(probabilities, 1)
            predicted_class = predicted_class_tensor.item()

            if self._should_filter_event(predicted_class, confidence.item(), frame_id): return None

            return InferenceResult(
                action_id=predicted_class,
                action_name=self.action_labels.get(predicted_class, f"Unknown ({predicted_class})"),
                confidence=confidence.item(), enhanced=bool(enhanced_actions),
                frame_id=frame_id, timestamp=timestamp, frame=frame
            )
        except Exception as e:
            logger.error(f"推理错误: {e}", exc_info=True)
            return None

    def _event_processing_worker(self):
        logger.info("事件处理工作线程已启动")
        while not self.stop_event.is_set():
            try:
                inference_result = self.inference_queue.get(timeout=1.0)
                if inference_result is None: break # 收到退出信号

                event = ActionEvent(
                    timestamp=inference_result.timestamp, action_id=inference_result.action_id,
                    action_name=inference_result.action_name, confidence=inference_result.confidence,
                    enhanced=inference_result.enhanced, frame_id=inference_result.frame_id
                )
                
                # FIX: 使用锁保护对last_event_frame的并发写操作
                with self.event_lock:
                    self.current_action_id = event.action_id
                    self.last_event_frame[event.action_id] = event.frame_id
                
                logger.info(f"事件触发. 动作 {event.action_id} 的last_event_frame更新为 {event.frame_id}.")

                if self.video_recorder and self.enable_video_recording:
                    try: self.video_queue.put_nowait({'type': 'start_recording', 'event': event, 'frame': inference_result.frame})
                    except queue.Full: logger.warning("视频队列已满，丢弃录制任务")
                    self.suppress_inference_until_ts = time.time() + 1.0
                else:
                    if self.event_filtering.get('duplicate_suppression', False):
                        if hasattr(self, 'llm_manager'):
                            try: self.llm_queue.put_nowait({'type': 'llm_inference', 'event': event})
                            except queue.Full: logger.warning("LLM队列已满，丢弃推理任务")
                        if self.event_callback: self.event_callback(event)
            except queue.Empty: continue
            except Exception as e: logger.error(f"事件处理错误: {e}", exc_info=True)
        logger.info("事件处理工作线程已停止")

    def _llm_inference_worker(self):
        logger.info("LLM推理工作线程已启动")
        while not self.stop_event.is_set():
            try:
                llm_task = self.llm_queue.get(timeout=1.0)
                if llm_task is None: break # 收到退出信号

                if llm_task.get('type') == 'llm_inference':
                    event = llm_task.get('event')
                    if hasattr(self, 'llm_manager') and event:
                        try:
                            llm_result = self.llm_manager.run_inference(event)
                            if llm_result and llm_result.success:
                                logger.info(f"事件 {event.action_name} 的LLM推理完成: {llm_result.result}")
                        except Exception as e: logger.error(f"LLM推理错误: {e}")
            except queue.Empty: continue
            except Exception as e: logger.error(f"LLM推理工作线程错误: {e}", exc_info=False)
        logger.info("LLM推理工作线程已停止")

    def _should_filter_event(self, action_id: int, confidence: float, frame_id: int) -> bool:
        if not self.event_filtering.get('enabled', False): return False
        if confidence < self.event_filtering.get('min_confidence', 0.1): return True
        
        if self.event_filtering.get('duplicate_suppression', False):
            # FIX: 使用锁保护对last_event_frame的并发读操作
            with self.event_lock:
                last_frame = self.last_event_frame.get(action_id)
            
            frame_window = self.event_filtering.get('duplicate_frame_window', 30)
            if (last_frame is not None and (frame_id - last_frame) < frame_window): return True
            
        return False
    
    def get_status(self) -> StreamStatus:
        return self.status
    
    def get_stats(self) -> Dict[str, Any]:
        # FIX: 使用锁来获取性能统计数据的线程安全快照
        with self.stats_lock:
            # 创建一个副本以避免在锁外修改
            stats_copy = self.performance_stats.copy()
            stats_copy['queue_sizes'] = {
                'frame_queue': self.frame_queue.qsize(), 'skeleton_queue': self.skeleton_queue.qsize(),
                'inference_queue': self.inference_queue.qsize(), 'video_queue': self.video_queue.qsize(),
                'llm_queue': self.llm_queue.qsize()
            }
        
        # FIX: 使用锁来获取事件数据的线程安全快照
        with self.event_lock:
            last_event_frame_copy = self.last_event_frame.copy()

        return {
            "status": self.status.value, "frame_count": self.frame_count,
            "last_event_frames": last_event_frame_copy, "target_actions": list(self.target_actions.keys()) if self.target_actions else [],
            "target_actions_config": self.target_actions, "base_boost_factor": getattr(self, 'base_boost_factor', 3.0),
            "priority_boost_factors": getattr(self, 'priority_boost_factors', {}), "dataset_type": getattr(self, 'dataset_type', 'ntu60'),
            "num_classes": getattr(self, 'num_classes', 60), "event_filtering": getattr(self, 'event_filtering', {}),
            "fps_target": getattr(self, 'fps_target', 30), "device": str(getattr(self, 'device', 'cpu')),
            "video_recording_enabled": getattr(self, 'enable_video_recording', False),
            "video_output_dir": getattr(self, 'video_recorder', None) and self.video_recorder.output_dir,
            "llm_inference_status": getattr(self, 'llm_manager', None) and self.llm_manager.get_status(),
            "performance_stats": stats_copy
        }

    def get_performance_summary(self) -> str:
        # 使用 get_stats() 以确保线程安全
        current_stats = self.get_stats()
        stats = current_stats.get('performance_stats', {})
        queue_sizes = stats.get('queue_sizes', {})

        return f"""
性能摘要:
==================
帧捕获FPS: {stats.get('frame_capture_fps', 0):.1f}
骨架检测FPS: {stats.get('skeleton_detection_fps', 0):.1f}
推理FPS: {stats.get('inference_fps', 0):.1f}
丢弃的帧: {stats.get('dropped_frames', 0)}

队列状态:
============
帧队列: {queue_sizes.get('frame_queue', 0)}
骨架队列: {queue_sizes.get('skeleton_queue', 0)}
推理队列: {queue_sizes.get('inference_queue', 0)}
视频队列: {queue_sizes.get('video_queue', 0)}
LLM队列: {queue_sizes.get('llm_queue', 0)}

系统状态:
=============
状态: {current_stats.get('status', 'N/A')}
总帧数: {current_stats.get('frame_count', 0)}
设备: {current_stats.get('device', 'N/A')}
视频录制: {'启用' if current_stats.get('video_recording_enabled', False) else '禁用'}
        """.strip()

    def _performance_monitor_worker(self):
        logger.info("性能监控工作线程已启动")
        monitor_interval = 5.0
        while not self.stop_event.wait(monitor_interval): # 等待5秒或直到stop_event被设置
            try:
                # FIX: 使用锁来安全地更新和读取统计数据
                with self.stats_lock:
                    self.performance_stats['queue_sizes'] = {
                        'frame_queue': self.frame_queue.qsize(), 
                        'skeleton_queue': self.skeleton_queue.qsize(),
                        'inference_queue': self.inference_queue.qsize(), 
                        'video_queue': self.video_queue.qsize(),
                        'llm_queue': self.llm_queue.qsize()
                    }
                    stats_for_log = self.performance_stats.copy()

                self._adaptive_performance_adjustment()

                if logger.isEnabledFor(logging.DEBUG): 
                    logger.debug(f"性能统计: {stats_for_log}")

            except Exception as e:
                logger.error(f"性能监控错误: {e}", exc_info=False)
                time.sleep(5.0) # 如果出错，等待一段时间再重试
        logger.info("性能监控工作线程已停止")

    def _adaptive_performance_adjustment(self):
        try:
            adaptive_config = self.config['stream_config'].get('performance', {}).get('adaptive_adjustment', {})
            if not adaptive_config.get('enabled', True):
                return
            
            with self.stats_lock: # FIX: 在锁内读取统计数据
                queue_sizes = self.performance_stats.get('queue_sizes', {})
                inference_fps = self.performance_stats.get('inference_fps', 0)

            self._last_adjustment_time = time.time()
            
            queue_thresholds = adaptive_config.get('queue_thresholds', {})
            high_load_ratio = queue_thresholds.get('high_load_ratio', 0.8)
            low_load_ratio = queue_thresholds.get('low_load_ratio', 0.3)
            
            frame_skipping_config = adaptive_config.get('frame_skipping', {})
            min_skip_frames = frame_skipping_config.get('min_skip_frames', 1)
            max_skip_frames = frame_skipping_config.get('max_skip_frames', 10)
            skip_adjustment_step = frame_skipping_config.get('adjustment_step', 1)

            frame_queue_ratio = queue_sizes.get('frame_queue', 0) / max(self.frame_queue.maxsize, 1)
            skeleton_queue_ratio = queue_sizes.get('skeleton_queue', 0) / max(self.skeleton_queue.maxsize, 1)

            current_max_skip = self.config['stream_config'].get('max_skip_frames', 3)
            if frame_queue_ratio > high_load_ratio or skeleton_queue_ratio > high_load_ratio:
                if current_max_skip < max_skip_frames:
                    new_skip_frames = min(current_max_skip + skip_adjustment_step, max_skip_frames)
                    self.config['stream_config']['max_skip_frames'] = new_skip_frames
                    logger.info(f"因队列积压，max_skip_frames增加到 {new_skip_frames}")
            elif frame_queue_ratio < low_load_ratio and skeleton_queue_ratio < low_load_ratio:
                if current_max_skip > min_skip_frames:
                    new_skip_frames = max(current_max_skip - skip_adjustment_step, min_skip_frames)
                    self.config['stream_config']['max_skip_frames'] = new_skip_frames
                    logger.info(f"因队列利用率低，max_skip_frames减少到 {new_skip_frames}")
        except Exception as e:
            logger.error(f"自适应性能调整错误: {e}")
