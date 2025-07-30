"""
不同数据集格式的标签提取器
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class VideoSample:
    """带标签信息的视频样本"""
    video_path: str
    action_id: int
    action_name: str
    metadata: Dict = None

class DatasetLabelExtractor(ABC):
    """数据集标签提取器的抽象基类"""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.action_mapping = self._load_action_mapping()

    @abstractmethod
    def _load_action_mapping(self) -> Dict[int, str]:
        """加载动作ID到名称的映射"""
        pass

    @abstractmethod
    def extract_samples(self) -> List[VideoSample]:
        """从数据集中提取所有带标签的视频样本"""
        pass

    @abstractmethod
    def extract_label_from_filename(self, filename: str) -> Tuple[int, str]:
        """从文件名中提取动作ID和名称"""
        pass

class NTURGBDExtractor(DatasetLabelExtractor):
    """NTU RGB+D数据集的标签提取器"""

    def _load_action_mapping(self) -> Dict[int, str]:
        """加载NTU RGB+D动作映射"""
        # NTU RGB+D 60个动作类别 (A1-A60)
        actions = {
            1: "drink water",
            2: "eat meal/snack", 
            3: "brushing teeth",
            4: "brushing hair",
            5: "drop",
            6: "pickup",
            7: "throw",
            8: "sitting down",
            9: "standing up (from sitting position)",
            10: "clapping",
            11: "reading",
            12: "writing",
            13: "tear up paper",
            14: "wear jacket",
            15: "take off jacket",
            16: "wear a shoe",
            17: "take off a shoe",
            18: "wear on glasses",
            19: "take off glasses",
            20: "put on a hat/cap",
            21: "take off a hat/cap",
            22: "cheer up",
            23: "hand waving",
            24: "kicking something",
            25: "reach into pocket",
            26: "hopping (one foot jumping)",
            27: "jump up",
            28: "make a phone call/answer phone",
            29: "playing with phone/tablet",
            30: "typing on a keyboard",
            31: "pointing to something with finger",
            32: "taking a selfie",
            33: "check time (from watch)",
            34: "rub two hands together",
            35: "nod head/bow",
            36: "shake head",
            37: "wipe face",
            38: "salute",
            39: "put the palms together",
            40: "cross hands in front (say stop)",
            41: "sneeze/cough",
            42: "staggering",
            43: "falling",
            44: "touch head (headache)",
            45: "touch chest (stomachache/heart pain)",
            46: "touch back (backache)",
            47: "touch neck (neckache)",
            48: "nausea or vomiting condition",
            49: "use a fan (with hand or paper)/feeling warm",
            50: "punching/slapping other person",
            51: "kicking other person",
            52: "pushing other person",
            53: "pat on back of other person",
            54: "point finger at the other person",
            55: "hugging other person",
            56: "giving something to other person",
            57: "touch other person's pocket",
            58: "handshaking",
            59: "walking towards each other",
            60: "walking apart from each other"
        }
        return actions
    
    def extract_label_from_filename(self, filename: str) -> Tuple[int, str]:
        """
        从NTU RGB+D文件名中提取动作ID
        格式: S017C003P020R002A059_rgb.avi
        """
        # 使用正则表达式从文件名中提取动作ID
        match = re.search(r'A(\d+)', filename)
        if match:
            action_id = int(match.group(1))
            action_name = self.action_mapping.get(action_id, f"未知动作 {action_id}")
            return action_id, action_name
        else:
            raise ValueError(f"无法从文件名中提取动作ID: {filename}")

    def extract_samples(self) -> List[VideoSample]:
        """提取所有NTU RGB+D视频样本"""
        samples = []

        if not os.path.exists(self.dataset_path):
            logger.error(f"数据集路径不存在: {self.dataset_path}")
            return samples

        # 遍历数据集目录中的所有文件
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(('.avi', '.mp4', '.mov')):
                    try:
                        action_id, action_name = self.extract_label_from_filename(file)
                        video_path = os.path.join(root, file)

                        sample = VideoSample(
                            video_path=video_path,
                            action_id=action_id,
                            action_name=action_name,
                            metadata={'dataset': 'ntu-rgbd', 'filename': file}
                        )
                        samples.append(sample)

                    except ValueError as e:
                        logger.warning(f"跳过文件 {file}: {e}")

        logger.info(f"从NTU RGB+D数据集中提取了 {len(samples)} 个样本")
        return samples

class AIGCExtractor(DatasetLabelExtractor):
    """AIGC数据集的标签提取器"""

    def _load_action_mapping(self) -> Dict[int, str]:
        """加载AIGC动作映射（使用NTU RGB+D映射）"""
        # AIGC数据集使用NTU RGB+D动作类别
        return NTURGBDExtractor(self.dataset_path)._load_action_mapping()

    def extract_label_from_filename(self, filename: str) -> Tuple[int, str]:
        """
        从AIGC元数据中提取动作ID
        此方法应该用元数据内容调用，而不是文件名
        """
        # 这将在extract_samples中为AIGC格式重写
        raise NotImplementedError("对于AIGC数据集请使用extract_samples")

    def _parse_metadata_file(self, metadata_path: str) -> Tuple[int, str]:
        """解析metadata.txt文件以提取动作信息"""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 从第一行提取动作标签
            # 格式: "动作标签: A47_touch_neck_neckache"
            lines = content.strip().split('\n')
            for line in lines:
                if line.startswith('动作标签:'):
                    label_part = line.split(':', 1)[1].strip()
                    # 从标签中提取动作ID，如"A47_touch_neck_neckache"
                    match = re.search(r'A(\d+)', label_part)
                    if match:
                        action_id = int(match.group(1))
                        action_name = self.action_mapping.get(action_id, f"未知动作 {action_id}")
                        return action_id, action_name

            raise ValueError(f"在元数据文件中找不到动作标签: {metadata_path}")

        except Exception as e:
            raise ValueError(f"解析元数据文件错误 {metadata_path}: {e}")
    
    def extract_samples(self) -> List[VideoSample]:
        """提取所有AIGC视频样本"""
        samples = []

        if not os.path.exists(self.dataset_path):
            logger.error(f"数据集路径不存在: {self.dataset_path}")
            return samples

        # 遍历AIGC数据集结构
        # 预期结构: datasets/aigc/A7_throw/01/video.mp4 + metadata.txt
        for root, dirs, files in os.walk(self.dataset_path):
            # 查找包含动作模式的目录（例如A7_throw）
            if 'metadata.txt' in files:
                # 在同一目录中查找视频文件
                video_file = None
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_file = file
                        break

                if video_file:
                    try:
                        metadata_path = os.path.join(root, 'metadata.txt')
                        action_id, action_name = self._parse_metadata_file(metadata_path)
                        video_path = os.path.join(root, video_file)

                        # 读取完整元数据以获取额外信息
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata_content = f.read()

                        sample = VideoSample(
                            video_path=video_path,
                            action_id=action_id,
                            action_name=action_name,
                            metadata={
                                'dataset': 'aigc',
                                'metadata_path': metadata_path,
                                'metadata_content': metadata_content,
                                'sample_dir': os.path.basename(root)
                            }
                        )
                        samples.append(sample)

                    except ValueError as e:
                        logger.warning(f"跳过样本 {root}: {e}")

        logger.info(f"从AIGC数据集中提取了 {len(samples)} 个样本")
        return samples

def create_extractor(dataset_type: str, dataset_path: str) -> DatasetLabelExtractor:
    """工厂函数，创建适当的数据集提取器"""
    if dataset_type == 'ntu-rgbd':
        return NTURGBDExtractor(dataset_path)
    elif dataset_type == 'aigc':
        return AIGCExtractor(dataset_path)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
