from utils.utils import DIRECT_MAPPING, yolo_to_ntu_mapping

from typing import Optional
import numpy as np
import torch
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class YOLOToNTUConverter:
    """Optimized YOLO to NTU skeleton converter"""

    def __init__(self, model_path: str = 'yolo11x-pose.pt'):
        self.model = YOLO(model_path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.direct_mapping = DIRECT_MAPPING
        self.ntu_keypoints = np.zeros((25, 3), dtype=np.float32)

    def extract_and_convert(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract YOLO keypoints and convert to NTU format in one step"""
        try:
            results = self.model(image, verbose=False)
            for result in results:
                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    yolo_kpts = result.keypoints.data[0].cpu().numpy()
                    return yolo_to_ntu_mapping(yolo_kpts, self.direct_mapping, self.ntu_keypoints)
            return None
        except Exception as e:
            logger.error(f"Error in keypoint extraction: {e}")
            return None