import sys
import traceback
import yaml
import logging
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import os
import re
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_env_variables():
    """加载环境变量"""
    # 尝试从.env文件加载环境变量
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"已加载环境变量文件: {env_path}")
    else:
        logger.warning(f"未找到.env文件: {env_path}")

def substitute_env_variables(config_str: str) -> str:
    """
    替换配置字符串中的环境变量占位符

    Args:
        config_str: 包含环境变量占位符的配置字符串

    Returns:
        替换后的配置字符串
    """
    def replace_env_var(match):
        var_name = match.group(1)
        env_value = os.getenv(var_name)
        if env_value is None:
            logger.warning(f"环境变量 {var_name} 未设置，保持原始占位符")
            return match.group(0)  # 返回原始占位符
        return env_value

    # 匹配 ${VAR_NAME} 格式的环境变量占位符
    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replace_env_var, config_str)

def import_class(import_str: str):
    """Import class dynamically from string."""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found ({traceback.format_exception(*sys.exc_info())})')

def load_config(config_path: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration from file with environment variable substitution or return default configuration."""
    if not config_path or not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return default_config

    try:
        # 加载环境变量
        load_env_variables()

        # 读取配置文件内容
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()

        # 替换环境变量占位符
        config_content = substitute_env_variables(config_content)

        # 解析YAML
        config = yaml.load(config_content, Loader=yaml.FullLoader)
        logger.info(f"Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        return default_config

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'stream_config': {
            'fps_target': 30,
            'frame_buffer_size': 30,
            'yolo_model_path': 'yolo11x-pose.pt',
            'skateformer_config_path': 'configs/ntu_yolo_pose/SkateFormer_j.yaml',
            'skateformer_weights_path': 'pretrained/ntu_yolo_pose/ntu-yolo-pose.pt',
            'window_size': 64,
            'confidence_threshold': 0.3,
            'min_keypoints': 20,
            'max_skip_frames': 3,
            'scene_change_threshold': 30.0,
            'target_actions': {
                'enabled': True,
                'boost_factor': 3.0,
                'actions': [
                    {'id': 0, 'name': 'drink water', 'priority': 'high'},
                    {'id': 7, 'name': 'sitting down', 'priority': 'high'},
                    {'id': 8, 'name': 'standing up', 'priority': 'high'},
                    {'id': 42, 'name': 'falling', 'priority': 'critical'}
                ]
            },
            'priority_boost_factors': {
                'critical': 5.0,
                'high': 3.0,
                'medium': 2.0,
                'low': 1.5
            },
            'event_filtering': {
                'enabled': True,
                'min_confidence': 0.1,
                'duplicate_suppression': True,
                'duplicate_time_window': 1.0
            },
            'performance': {
                'use_gpu': True,
                'gpu_memory_fraction': 0.8,
                'batch_size': 1,
                'num_workers': 2,
                'adaptive_adjustment': {
                    'enabled': True,
                    'queue_thresholds': {
                        'high_load_ratio': 0.8,
                        'low_load_ratio': 0.3
                    },
                    'frame_skipping': {
                        'min_skip_frames': 1,
                        'max_skip_frames': 10,
                        'adjustment_step': 1
                    },
                    'scene_change': {
                        'min_threshold': 30.0,
                        'max_threshold': 50.0,
                        'adjustment_step': 5.0,
                        'low_fps_trigger': 10
                    },
                    'monitoring': {
                        'avg_processing_time_warning': 0.1
                    }
                }
            },
            'rtsp': {
                'buffer_size': 1,
                'timeout': 5000,
                'reconnect_attempts': 3,
                'reconnect_delay': 5
            },
            'video_recording': {
                'output_dir': 'event_clips',
                'pre_event_seconds': 5.0,
                'post_event_seconds': 5.0,
                'max_buffer_size': 300,
                'auto_cleanup_days': 30
            }
        }
    }

def apply_partition(data_numpy: np.ndarray) -> np.ndarray:
    """Apply joint partitioning to keep 24 joints (excluding joint 21)"""
    right_arm = np.array([6, 7, 21, 22])
    left_arm = np.array([10, 11, 23, 24])
    right_leg = np.array([12, 13, 14, 15])
    left_leg = np.array([16, 17, 18, 19])
    h_torso = np.array([4, 8, 5, 9])
    w_torso = np.array([1, 2, 0, 3])
    partition_indices = np.concatenate((right_arm, left_arm, right_leg, left_leg, h_torso, w_torso), axis=-1)
    return data_numpy[:, :, partition_indices, :]

def extend_extremities(ntu_keypoints: np.ndarray, yolo_keypoints: np.ndarray):
    """Extend hand and foot points according to NTU structure"""
    if len(yolo_keypoints) > 9 and yolo_keypoints[9][2] > 0.3:
        ntu_keypoints[7] = yolo_keypoints[9]
        ntu_keypoints[21] = yolo_keypoints[9] + np.array([15, 0, 0])
        ntu_keypoints[22] = yolo_keypoints[9] + np.array([10, -5, 0])
    if len(yolo_keypoints) > 10 and yolo_keypoints[10][2] > 0.3:
        ntu_keypoints[11] = yolo_keypoints[10]
        ntu_keypoints[23] = yolo_keypoints[10] + np.array([-15, 0, 0])
        ntu_keypoints[24] = yolo_keypoints[10] + np.array([-10, -5, 0])
    if len(yolo_keypoints) > 15 and yolo_keypoints[15][2] > 0.3:
        ntu_keypoints[15] = yolo_keypoints[15]
    if len(yolo_keypoints) > 16 and yolo_keypoints[16][2] > 0.3:
        ntu_keypoints[19] = yolo_keypoints[16]

def yolo_to_ntu_mapping(yolo_keypoints: np.ndarray, direct_mapping: Dict[int, int], ntu_keypoints: np.ndarray) -> np.ndarray:
    """Optimized YOLO to NTU mapping"""
    ntu_keypoints.fill(0)
    for yolo_idx, ntu_idx in direct_mapping.items():
        if yolo_idx < len(yolo_keypoints):
            ntu_keypoints[ntu_idx] = yolo_keypoints[yolo_idx]
    if len(yolo_keypoints) > 12:
        if yolo_keypoints[11][2] > 0.3 and yolo_keypoints[12][2] > 0.3:
            ntu_keypoints[0] = (yolo_keypoints[11] + yolo_keypoints[12]) * 0.5
        if yolo_keypoints[5][2] > 0.3 and yolo_keypoints[6][2] > 0.3:
            ntu_keypoints[2] = (yolo_keypoints[5] + yolo_keypoints[6]) * 0.5
        if ntu_keypoints[0][2] > 0.3 and ntu_keypoints[2][2] > 0.3:
            ntu_keypoints[1] = (ntu_keypoints[0] + ntu_keypoints[2]) * 0.5
        if ntu_keypoints[0][2] > 0.3 and ntu_keypoints[1][2] > 0.3:
            ntu_keypoints[20] = (ntu_keypoints[0] + ntu_keypoints[1]) * 0.5
    extend_extremities(ntu_keypoints, yolo_keypoints)
    return ntu_keypoints.copy()

DIRECT_MAPPING = {
    0: 3, 5: 4, 6: 8, 7: 5, 8: 9, 9: 6, 10: 10, 11: 12, 12: 16,
    13: 13, 14: 17, 15: 14, 16: 18
}

def create_metadata_file(recording_info: Dict, fps: int, pre_event_seconds: float, post_event_seconds: float):
    """Create metadata file for the video clip"""
    metadata = {
        'event': {
            'timestamp': recording_info['event'].timestamp,
            'action_id': recording_info['event'].action_id,
            'action_name': recording_info['event'].action_name,
            'confidence': recording_info['event'].confidence,
            'enhanced': recording_info['event'].enhanced,
            'frame_id': recording_info['event'].frame_id
        },
        'recording': {
            'output_path': recording_info['output_path'],
            'frames_written': recording_info['frames_written'],
            'fps': fps,
            'pre_event_seconds': pre_event_seconds,
            'post_event_seconds': post_event_seconds
        },
        'created_at': datetime.now().isoformat()
    }
    metadata_path = Path(recording_info['output_path']).with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_action_labels(dataset_type: str) -> Dict[int, str]:
    """Load action labels based on dataset type"""
    labels = {
        0: "drink water", 1: "eat meal/snack", 2: "brushing teeth", 3: "brushing hair",
        4: "drop", 5: "pickup", 6: "throw", 7: "sitting down", 8: "standing up",
        9: "clapping", 10: "reading", 11: "writing", 12: "tear up paper",
        13: "wear jacket", 14: "take off jacket", 15: "wear a shoe", 16: "take off a shoe",
        17: "wear on glasses", 18: "take off glasses", 19: "put on a hat/cap", 20: "take off a hat/cap",
        21: "cheer up", 22: "hand waving", 23: "kicking something", 24: "reach into pocket",
        25: "hopping", 26: "jump up", 27: "make a phone call", 28: "playing with phone/tablet",
        29: "typing on a keyboard", 30: "pointing to something", 31: "taking a selfie",
        32: "check time", 33: "rub two hands together", 34: "nod head/bow", 35: "shake head",
        36: "wipe face", 37: "salute", 38: "put palms together", 39: "cross hands in front",
        40: "sneeze/cough", 41: "staggering", 42: "falling", 43: "touch head",
        44: "touch chest", 45: "touch back", 46: "touch neck", 47: "nausea or vomiting",
        48: "use a fan", 49: "punching/slapping", 50: "kicking other person",
        51: "pushing other person", 52: "pat on back", 53: "point finger at other person",
        54: "hugging other person", 55: "giving something", 56: "touch other person's pocket",
        57: "handshaking", 58: "walking towards each other", 59: "walking apart"
    }
    if dataset_type == "ntu120":
        ntu120_additional = {
            60: "put on headphone", 61: "take off headphone", 62: "shoot at basket",
            63: "bounce ball", 64: "tennis bat swing", 65: "juggling table tennis balls",
            66: "hush", 67: "flick hair", 68: "thumb up", 69: "thumb down",
            70: "make ok sign", 71: "make victory sign", 72: "staple book",
            73: "counting money", 74: "cutting nails", 75: "cutting paper",
            76: "snapping fingers", 77: "open bottle", 78: "sniff", 79: "squat down",
            80: "toss a coin", 81: "fold paper", 82: "ball up paper", 83: "play magic cube",
            84: "apply cream on face", 85: "apply cream on hand", 86: "put on bag",
            87: "take off bag", 88: "put something into bag", 89: "take something out of bag",
            90: "open a box", 91: "move heavy objects", 92: "shake fist", 93: "throw up cap",
            94: "hands up", 95: "cross arms", 96: "arm circles", 97: "arm swings",
            98: "running on spot", 99: "butt kicks", 100: "cross toe touch", 101: "side kick",
            102: "yawn", 103: "stretch oneself", 104: "blow nose", 105: "hit other person",
            106: "wield knife", 107: "knock over other person", 108: "grab other person's stuff",
            109: "shoot at other person", 110: "step on foot", 111: "high-five",
            112: "cheers and drink", 113: "carry something together", 114: "take a photo",
            115: "follow other person", 116: "whisper in ear", 117: "exchange things",
            118: "support somebody", 119: "finger-guessing game"
        }
        labels.update(ntu120_additional)
    return labels
