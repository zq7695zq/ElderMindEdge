"""
动作识别系统的评估指标
"""

import time
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging

from .dataset_extractors import VideoSample
from .inference_modes import InferenceResult

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    """延迟测量指标"""
    total_samples: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def mean_latency(self) -> float:
        """计算平均延迟"""
        return self.total_latency / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def median_latency(self) -> float:
        """计算中位延迟"""
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def std_latency(self) -> float:
        """计算延迟的标准差"""
        return statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0.0

    def add_measurement(self, latency: float):
        """添加延迟测量"""
        self.total_samples += 1
        self.total_latency += latency
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)
        self.latencies.append(latency)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于报告"""
        return {
            'total_samples': self.total_samples,
            'mean_latency': self.mean_latency,
            'median_latency': self.median_latency,
            'min_latency': self.min_latency if self.min_latency != float('inf') else 0.0,
            'max_latency': self.max_latency,
            'std_latency': self.std_latency
        }

@dataclass
class AccuracyMetrics:
    """准确率测量指标"""
    total_samples: int = 0
    correct_predictions: int = 0
    predictions_by_class: Dict[int, int] = field(default_factory=dict)
    correct_by_class: Dict[int, int] = field(default_factory=dict)
    confusion_matrix: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @property
    def overall_accuracy(self) -> float:
        """计算总体准确率"""
        return self.correct_predictions / self.total_samples if self.total_samples > 0 else 0.0

    def add_prediction(self, true_label: int, predicted_label: int):
        """添加预测结果"""
        self.total_samples += 1

        # 更新总体准确率
        if true_label == predicted_label:
            self.correct_predictions += 1

        # 更新每类统计
        if predicted_label not in self.predictions_by_class:
            self.predictions_by_class[predicted_label] = 0
        self.predictions_by_class[predicted_label] += 1

        if true_label == predicted_label:
            if true_label not in self.correct_by_class:
                self.correct_by_class[true_label] = 0
            self.correct_by_class[true_label] += 1

        # 更新混淆矩阵
        key = (true_label, predicted_label)
        if key not in self.confusion_matrix:
            self.confusion_matrix[key] = 0
        self.confusion_matrix[key] += 1

    def get_class_accuracy(self, class_id: int) -> float:
        """获取特定类别的准确率"""
        correct = self.correct_by_class.get(class_id, 0)
        total = sum(1 for (true_label, _) in self.confusion_matrix.keys() if true_label == class_id)
        return correct / total if total > 0 else 0.0

    def get_precision(self, class_id: int) -> float:
        """获取特定类别的精确率"""
        correct = self.correct_by_class.get(class_id, 0)
        predicted = self.predictions_by_class.get(class_id, 0)
        return correct / predicted if predicted > 0 else 0.0

    def get_recall(self, class_id: int) -> float:
        """获取特定类别的召回率"""
        return self.get_class_accuracy(class_id)

    def get_f1_score(self, class_id: int) -> float:
        """获取特定类别的F1分数"""
        precision = self.get_precision(class_id)
        recall = self.get_recall(class_id)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于报告"""
        # 计算每类指标
        class_metrics = {}
        all_classes = set()
        all_classes.update(self.predictions_by_class.keys())
        all_classes.update(self.correct_by_class.keys())
        all_classes.update(true_label for (true_label, _) in self.confusion_matrix.keys())

        for class_id in all_classes:
            class_metrics[class_id] = {
                'accuracy': self.get_class_accuracy(class_id),
                'precision': self.get_precision(class_id),
                'recall': self.get_recall(class_id),
                'f1_score': self.get_f1_score(class_id)
            }

        return {
            'total_samples': self.total_samples,
            'correct_predictions': self.correct_predictions,
            'overall_accuracy': self.overall_accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': {f"{k[0]}->{k[1]}": v for k, v in self.confusion_matrix.items()}
        }

class EvaluationMetrics:
    """主要评估指标容器"""

    def __init__(self):
        self.latency_metrics = LatencyMetrics()
        self.accuracy_metrics = AccuracyMetrics()
        self.start_time = None
        self.end_time = None
        self.inference_mode = None
        self.dataset_info = {}

    def start_evaluation(self, inference_mode: str, dataset_info: Dict[str, Any]):
        """开始评估计时"""
        self.start_time = time.time()
        self.inference_mode = inference_mode
        self.dataset_info = dataset_info
        logger.info(f"使用 {inference_mode} 推理模式开始评估")

    def end_evaluation(self):
        """结束评估计时"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        logger.info(f"评估在 {duration:.2f} 秒内完成")
    
    def add_result(self, sample: VideoSample, results: List[InferenceResult]):
        """为样本添加评估结果"""
        if not results:
            logger.warning(f"样本无结果: {sample.video_path}")
            return

        # 对于准确率，使用最佳结果（最高置信度）
        best_result = max(results, key=lambda r: r.confidence)

        # 添加准确率测量
        self.accuracy_metrics.add_prediction(sample.action_id, best_result.action_id)

        # 添加延迟测量（使用第一个结果的延迟）
        first_result = results[0]
        self.latency_metrics.add_measurement(first_result.latency)

        logger.debug(f"为 {sample.video_path} 添加结果: "
                    f"真实={sample.action_id}, 预测={best_result.action_id}, "
                    f"置信度={best_result.confidence:.3f}, 延迟={first_result.latency:.3f}秒")

    def get_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        duration = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0

        summary = {
            'evaluation_info': {
                'inference_mode': self.inference_mode,
                'dataset_info': self.dataset_info,
                'evaluation_duration': duration,
                'start_time': self.start_time,
                'end_time': self.end_time
            },
            'latency_metrics': self.latency_metrics.to_dict(),
            'accuracy_metrics': self.accuracy_metrics.to_dict()
        }

        return summary
    
    def print_summary(self):
        """打印评估摘要到控制台"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("评估摘要")
        print("="*60)

        # 评估信息
        eval_info = summary['evaluation_info']
        print(f"推理模式: {eval_info['inference_mode']}")
        print(f"数据集: {eval_info['dataset_info'].get('type', 'unknown')}")
        print(f"总样本数: {eval_info['dataset_info'].get('total_samples', 0)}")
        print(f"评估时长: {eval_info['evaluation_duration']:.2f} 秒")

        # 延迟指标
        print("\n延迟指标:")
        print("-" * 30)
        latency = summary['latency_metrics']
        print(f"平均延迟: {latency['mean_latency']:.3f} 秒")
        print(f"中位延迟: {latency['median_latency']:.3f} 秒")
        print(f"最小延迟: {latency['min_latency']:.3f} 秒")
        print(f"最大延迟: {latency['max_latency']:.3f} 秒")
        print(f"标准差: {latency['std_latency']:.3f} 秒")

        # 准确率指标
        print("\n准确率指标:")
        print("-" * 30)
        accuracy = summary['accuracy_metrics']
        print(f"总体准确率: {accuracy['overall_accuracy']:.3f} ({accuracy['overall_accuracy']*100:.1f}%)")
        print(f"正确预测: {accuracy['correct_predictions']}/{accuracy['total_samples']}")

        # 表现最佳的类别
        if accuracy['class_metrics']:
            print("\n表现最佳的类别:")
            print("-" * 30)
            class_metrics = accuracy['class_metrics']
            sorted_classes = sorted(class_metrics.items(),
                                  key=lambda x: x[1]['f1_score'], reverse=True)

            for i, (class_id, metrics) in enumerate(sorted_classes[:5]):
                print(f"类别 {class_id}: F1={metrics['f1_score']:.3f}, "
                      f"准确率={metrics['accuracy']:.3f}, "
                      f"精确率={metrics['precision']:.3f}, "
                      f"召回率={metrics['recall']:.3f}")

        print("="*60)

def calculate_metrics(samples: List[VideoSample],
                     all_results: List[List[InferenceResult]],
                     inference_mode: str) -> EvaluationMetrics:
    """从样本和结果计算评估指标"""
    metrics = EvaluationMetrics()

    dataset_info = {
        'type': samples[0].metadata.get('dataset', 'unknown') if samples else 'unknown',
        'total_samples': len(samples)
    }

    metrics.start_evaluation(inference_mode, dataset_info)

    for sample, results in zip(samples, all_results):
        metrics.add_result(sample, results)

    metrics.end_evaluation()

    return metrics
