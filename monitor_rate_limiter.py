#!/usr/bin/env python3
"""
LLM限流器监控脚本
实时监控限流器状态，用于生产环境监控
"""
import time
import json
import logging
from datetime import datetime
from utils.llm_utils import LLMInferenceManager
from utils.utils import load_config, get_default_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RateLimiterMonitor:
    """限流器监控器"""
    
    def __init__(self, config_path="configs/stream_config.yaml"):
        self.config_path = config_path
        self.llm_manager = None
        self.monitoring = False
        self.stats = {
            'start_time': None,
            'total_requests': 0,
            'successful_requests': 0,
            'rate_limited_requests': 0,
            'peak_concurrent': 0,
            'peak_per_minute': 0
        }
    
    def initialize(self):
        """初始化监控器"""
        try:
            # 加载配置
            config = load_config(self.config_path, get_default_config())
            llm_config = config['stream_config'].get('llm_inference', {})
            
            # 创建LLM管理器
            self.llm_manager = LLMInferenceManager(llm_config)
            
            if not self.llm_manager.initialize():
                logger.error("LLM管理器初始化失败")
                return False
            
            self.stats['start_time'] = datetime.now()
            logger.info("限流器监控器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"监控器初始化失败: {e}")
            return False
    
    def get_current_status(self):
        """获取当前状态"""
        if not self.llm_manager:
            return None
        
        status = self.llm_manager.get_status()
        rate_limiter_status = status.get('rate_limiter', {})
        
        # 更新统计信息
        if rate_limiter_status.get('enabled'):
            concurrent = rate_limiter_status.get('concurrent', {})
            rate_limits = rate_limiter_status.get('rate_limits', {})
            
            current_concurrent = concurrent.get('current', 0)
            current_per_minute = rate_limits.get('per_minute', {}).get('current', 0)
            
            self.stats['peak_concurrent'] = max(self.stats['peak_concurrent'], current_concurrent)
            self.stats['peak_per_minute'] = max(self.stats['peak_per_minute'], current_per_minute)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'llm_status': status,
            'monitor_stats': self.stats.copy()
        }
    
    def print_status(self, status):
        """打印状态信息"""
        if not status:
            print("无法获取状态信息")
            return
        
        llm_status = status['llm_status']
        rate_limiter = llm_status.get('rate_limiter', {})
        monitor_stats = status['monitor_stats']
        
        print(f"\n{'='*60}")
        print(f"时间: {status['timestamp']}")
        print(f"{'='*60}")
        
        # LLM状态
        print(f"LLM推理状态:")
        print(f"  启用: {llm_status.get('enabled', False)}")
        print(f"  模式: {llm_status.get('mode', 'N/A')}")
        print(f"  初始化: {llm_status.get('initialized', False)}")
        
        # 限流器状态
        if rate_limiter.get('enabled'):
            concurrent = rate_limiter.get('concurrent', {})
            rate_limits = rate_limiter.get('rate_limits', {})
            config = rate_limiter.get('config', {})
            
            print(f"\n限流器状态:")
            print(f"  并发控制:")
            print(f"    当前: {concurrent.get('current', 0)}")
            print(f"    最大: {concurrent.get('max', 0)}")
            print(f"    可用: {concurrent.get('available', 0)}")
            
            print(f"  频率限制:")
            per_minute = rate_limits.get('per_minute', {})
            per_hour = rate_limits.get('per_hour', {})
            
            print(f"    每分钟: {per_minute.get('current', 0)}/{per_minute.get('max', 0)} "
                  f"(剩余: {per_minute.get('remaining', 0)})")
            print(f"    每小时: {per_hour.get('current', 0)}/{per_hour.get('max', 0)} "
                  f"(剩余: {per_hour.get('remaining', 0)})")
            
            print(f"  配置:")
            print(f"    队列超时: {config.get('queue_timeout', 0)}秒")
            print(f"    重试延迟: {config.get('retry_delay', 0)}秒")
        else:
            print(f"\n限流器状态: 已禁用")
        
        # 监控统计
        if monitor_stats['start_time']:
            runtime = datetime.now() - monitor_stats['start_time']
            print(f"\n监控统计:")
            print(f"  运行时间: {runtime}")
            print(f"  峰值并发: {monitor_stats['peak_concurrent']}")
            print(f"  峰值每分钟: {monitor_stats['peak_per_minute']}")
    
    def save_status_to_file(self, status, filename="rate_limiter_status.json"):
        """保存状态到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")
    
    def start_monitoring(self, interval=10, save_to_file=False):
        """开始监控"""
        if not self.initialize():
            return
        
        self.monitoring = True
        logger.info(f"开始监控限流器状态，间隔: {interval}秒")
        
        try:
            while self.monitoring:
                status = self.get_current_status()
                
                # 打印状态
                self.print_status(status)
                
                # 保存到文件
                if save_to_file and status:
                    self.save_status_to_file(status)
                
                # 等待下一次检查
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止监控")
        except Exception as e:
            logger.error(f"监控过程中出错: {e}")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.llm_manager:
            self.llm_manager.cleanup()
        logger.info("监控已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM限流器监控工具')
    parser.add_argument('--config', '-c', default='configs/stream_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--interval', '-i', type=int, default=10,
                       help='监控间隔(秒)')
    parser.add_argument('--save-file', '-s', action='store_true',
                       help='保存状态到文件')
    parser.add_argument('--once', action='store_true',
                       help='只检查一次状态')
    
    args = parser.parse_args()
    
    monitor = RateLimiterMonitor(args.config)
    
    if args.once:
        # 只检查一次
        if monitor.initialize():
            status = monitor.get_current_status()
            monitor.print_status(status)
            if args.save_file:
                monitor.save_status_to_file(status)
            monitor.stop_monitoring()
    else:
        # 持续监控
        monitor.start_monitoring(args.interval, args.save_file)


if __name__ == "__main__":
    main()
