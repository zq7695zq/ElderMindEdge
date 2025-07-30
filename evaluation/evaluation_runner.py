"""
动作识别系统的主要评估运行器
"""

import os
import json
import time
import logging
from typing import Dict, Any, List
from datetime import datetime

from utils.utils import load_config, get_default_config
from .dataset_extractors import create_extractor, VideoSample
from .inference_modes import create_inference_mode, InferenceResult
from .metrics import EvaluationMetrics, calculate_metrics

logger = logging.getLogger(__name__)

class EvaluationRunner:
    """主要评估运行器"""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path or "configs/stream_config.yaml", get_default_config())
        self.results_dir = None
        
    def run_evaluation(self,
                      dataset_path: str,
                      dataset_type: str,
                      inference_mode: str,
                      output_dir: str = "evaluation_results") -> EvaluationMetrics:
        """
        在数据集上运行完整评估

        Args:
            dataset_path: 数据集目录路径
            dataset_type: 数据集类型 ('ntu-rgbd' 或 'aigc')
            inference_mode: 推理模式 ('local', 'cloud', 或 'hybrid')
            output_dir: 结果输出目录

        Returns:
            包含结果的EvaluationMetrics对象
        """

        # 设置输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"{dataset_type}_{inference_mode}_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info(f"开始评估: {dataset_type} 数据集使用 {inference_mode} 推理")
        logger.info(f"结果将保存到: {self.results_dir}")

        try:
            # 步骤1: 提取数据集样本
            logger.info("步骤1: 提取数据集样本...")
            extractor = create_extractor(dataset_type, dataset_path)
            samples = extractor.extract_samples()

            if not samples:
                logger.error("数据集中未找到样本")
                return None

            logger.info(f"找到 {len(samples)} 个样本")

            # 步骤2: 设置推理模式
            logger.info(f"步骤2: 设置 {inference_mode} 推理模式...")
            inference_handler = create_inference_mode(inference_mode, self.config)

            if not inference_handler.setup():
                logger.error("设置推理模式失败")
                return None

            # 步骤3: 处理样本
            logger.info("步骤3: 处理样本...")
            all_results = []

            for i, sample in enumerate(samples):
                logger.info(f"处理样本 {i+1}/{len(samples)}: {os.path.basename(sample.video_path)}")

                try:
                    # 处理视频
                    results = inference_handler.process_video(sample.video_path)
                    all_results.append(results)

                    logger.info(f"样本 {i+1} 完成: 生成了 {len(results)} 个结果")

                except Exception as e:
                    logger.error(f"处理样本 {i+1} 时出错: {e}")
                    all_results.append([])  # 为失败的样本添加空结果

            # 步骤4: 计算指标
            logger.info("步骤4: 计算指标...")
            metrics = calculate_metrics(samples, all_results, inference_mode)

            # 步骤5: 保存结果
            logger.info("步骤5: 保存结果...")
            self._save_results(samples, all_results, metrics)

            # 步骤6: 打印摘要
            metrics.print_summary()

            return metrics

        except Exception as e:
            logger.error(f"评估失败: {e}")
            return None

        finally:
            # 清理
            if 'inference_handler' in locals():
                inference_handler.cleanup()
    
    def _save_results(self,
                     samples: List[VideoSample],
                     all_results: List[List[InferenceResult]],
                     metrics: EvaluationMetrics):
        """保存评估结果到文件"""

        # 保存详细结果
        detailed_results = []
        for sample, results in zip(samples, all_results):
            sample_result = {
                'video_path': sample.video_path,
                'true_action_id': sample.action_id,
                'true_action_name': sample.action_name,
                'metadata': sample.metadata,
                'inference_results': [
                    {
                        'action_id': r.action_id,
                        'action_name': r.action_name,
                        'confidence': r.confidence,
                        'latency': r.latency,
                        'inference_time': r.inference_time,
                        'frame_timestamp': r.frame_timestamp,
                        'enhanced': r.enhanced,
                        'source': r.source
                    } for r in results
                ]
            }
            detailed_results.append(sample_result)

        # 保存详细结果为JSON
        detailed_path = os.path.join(self.results_dir, "detailed_results.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        # 保存指标摘要
        summary_path = os.path.join(self.results_dir, "metrics_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(metrics.get_summary(), f, indent=2, ensure_ascii=False)

        # 保存人类可读的报告
        report_path = os.path.join(self.results_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            self._write_text_report(f, metrics, len(samples))

        logger.info(f"结果已保存到 {self.results_dir}")
        logger.info(f"  - 详细结果: {detailed_path}")
        logger.info(f"  - 指标摘要: {summary_path}")
        logger.info(f"  - 文本报告: {report_path}")
    
    def _write_text_report(self, file, metrics: EvaluationMetrics, total_samples: int):
        """写入人类可读的文本报告"""
        summary = metrics.get_summary()

        file.write("动作识别评估报告\n")
        file.write("=" * 50 + "\n\n")

        # 评估信息
        eval_info = summary['evaluation_info']
        file.write("评估配置\n")
        file.write("-" * 25 + "\n")
        file.write(f"推理模式: {eval_info['inference_mode']}\n")
        file.write(f"数据集类型: {eval_info['dataset_info'].get('type', 'unknown')}\n")
        file.write(f"总样本数: {total_samples}\n")
        file.write(f"评估时长: {eval_info['evaluation_duration']:.2f} 秒\n")
        file.write(f"开始时间: {datetime.fromtimestamp(eval_info['start_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"结束时间: {datetime.fromtimestamp(eval_info['end_time']).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 延迟指标
        file.write("延迟性能\n")
        file.write("-" * 20 + "\n")
        latency = summary['latency_metrics']
        file.write(f"平均延迟: {latency['mean_latency']:.3f} 秒\n")
        file.write(f"中位延迟: {latency['median_latency']:.3f} 秒\n")
        file.write(f"最小延迟: {latency['min_latency']:.3f} 秒\n")
        file.write(f"最大延迟: {latency['max_latency']:.3f} 秒\n")
        file.write(f"标准差: {latency['std_latency']:.3f} 秒\n\n")

        # 准确率指标
        file.write("准确率性能\n")
        file.write("-" * 20 + "\n")
        accuracy = summary['accuracy_metrics']
        file.write(f"总体准确率: {accuracy['overall_accuracy']:.3f} ({accuracy['overall_accuracy']*100:.1f}%)\n")
        file.write(f"正确预测: {accuracy['correct_predictions']}/{accuracy['total_samples']}\n\n")

        # 每类性能
        if accuracy['class_metrics']:
            file.write("每类性能\n")
            file.write("-" * 22 + "\n")
            file.write(f"{'类别':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}\n")
            file.write("-" * 50 + "\n")

            class_metrics = accuracy['class_metrics']
            sorted_classes = sorted(class_metrics.items(), key=lambda x: int(x[0]))

            for class_id, metrics_dict in sorted_classes:
                file.write(f"{class_id:<8} {metrics_dict['accuracy']:<10.3f} "
                          f"{metrics_dict['precision']:<10.3f} {metrics_dict['recall']:<10.3f} "
                          f"{metrics_dict['f1_score']:<10.3f}\n")

        file.write("\n" + "=" * 50 + "\n")

def run_evaluation(args):
    """命令行评估的主要入口点"""

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("开始动作识别评估")

    # 验证输入
    if not os.path.exists(args.dataset_path):
        logger.error(f"数据集路径不存在: {args.dataset_path}")
        return

    # 创建评估运行器
    runner = EvaluationRunner(args.config)

    # 运行评估
    metrics = runner.run_evaluation(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        inference_mode=args.inference_mode,
        output_dir=args.output_dir
    )

    if metrics:
        logger.info("评估成功完成")
    else:
        logger.error("评估失败")
        return 1

    return 0
