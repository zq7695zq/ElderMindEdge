"""
LLM推理限流器 - 控制LLM API的并发请求和频率限制
"""
import time
import threading
import logging
from typing import Dict, Any, Optional
from collections import deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """限流异常"""
    pass


class LLMRateLimiter:
    """LLM推理限流器
    
    功能：
    1. 控制最大并发请求数
    2. 控制每分钟/每小时的请求频率
    3. 提供队列等待机制
    4. 支持重试延迟
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # 并发控制
        self.max_concurrent = config.get('max_concurrent_requests', 3)
        self.current_concurrent = 0
        self.concurrent_lock = threading.Lock()
        self.concurrent_semaphore = threading.Semaphore(self.max_concurrent)
        
        # 频率控制
        self.max_per_minute = config.get('max_requests_per_minute', 30)
        self.max_per_hour = config.get('max_requests_per_hour', 500)
        
        # 请求历史记录 (时间戳队列)
        self.minute_requests = deque()  # 最近一分钟的请求
        self.hour_requests = deque()    # 最近一小时的请求
        self.history_lock = threading.Lock()
        
        # 队列和重试配置
        self.queue_timeout = config.get('queue_timeout', 30)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        logger.info(f"LLM限流器初始化: 并发={self.max_concurrent}, "
                   f"分钟限制={self.max_per_minute}, 小时限制={self.max_per_hour}")
    
    def _cleanup_old_requests(self):
        """清理过期的请求记录"""
        current_time = time.time()
        
        # 清理一分钟前的记录
        while self.minute_requests and current_time - self.minute_requests[0] > 60:
            self.minute_requests.popleft()
        
        # 清理一小时前的记录
        while self.hour_requests and current_time - self.hour_requests[0] > 3600:
            self.hour_requests.popleft()
    
    def _can_make_request(self) -> tuple[bool, str]:
        """检查是否可以发起请求
        
        Returns:
            (can_request, reason): 是否可以请求和原因
        """
        if not self.enabled:
            return True, "限流器已禁用"
        
        current_time = time.time()
        
        with self.history_lock:
            self._cleanup_old_requests()
            
            # 检查每分钟限制
            if len(self.minute_requests) >= self.max_per_minute:
                return False, f"每分钟请求数已达上限 ({self.max_per_minute})"
            
            # 检查每小时限制
            if len(self.hour_requests) >= self.max_per_hour:
                return False, f"每小时请求数已达上限 ({self.max_per_hour})"
        
        return True, "可以发起请求"
    
    def _record_request(self):
        """记录一次请求"""
        current_time = time.time()
        
        with self.history_lock:
            self.minute_requests.append(current_time)
            self.hour_requests.append(current_time)
            self._cleanup_old_requests()
    
    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """获取请求许可的上下文管理器
        
        Args:
            timeout: 超时时间，None表示使用默认配置
            
        Raises:
            RateLimitExceeded: 当超过限流限制时
        """
        if not self.enabled:
            # 限流器禁用时直接通过
            yield
            return
        
        if timeout is None:
            timeout = self.queue_timeout
        
        start_time = time.time()
        acquired = False
        
        try:
            # 尝试获取并发信号量
            if not self.concurrent_semaphore.acquire(timeout=timeout):
                raise RateLimitExceeded(f"等待并发许可超时 ({timeout}秒)")
            
            acquired = True
            
            # 检查频率限制
            while True:
                can_request, reason = self._can_make_request()
                if can_request:
                    break
                
                # 检查是否超时
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise RateLimitExceeded(f"等待频率限制解除超时: {reason}")
                
                # 等待重试延迟
                logger.debug(f"频率限制中，等待 {self.retry_delay}秒: {reason}")
                time.sleep(self.retry_delay)
            
            # 更新并发计数
            with self.concurrent_lock:
                self.current_concurrent += 1
            
            # 记录请求
            self._record_request()
            
            logger.debug(f"获取LLM请求许可成功，当前并发: {self.current_concurrent}/{self.max_concurrent}")
            
            yield
            
        finally:
            if acquired:
                # 释放并发信号量
                self.concurrent_semaphore.release()
                
                # 更新并发计数
                with self.concurrent_lock:
                    self.current_concurrent = max(0, self.current_concurrent - 1)
                
                logger.debug(f"释放LLM请求许可，当前并发: {self.current_concurrent}/{self.max_concurrent}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取限流器状态"""
        with self.history_lock:
            self._cleanup_old_requests()
            minute_count = len(self.minute_requests)
            hour_count = len(self.hour_requests)
        
        with self.concurrent_lock:
            current_concurrent = self.current_concurrent
        
        return {
            'enabled': self.enabled,
            'concurrent': {
                'current': current_concurrent,
                'max': self.max_concurrent,
                'available': self.max_concurrent - current_concurrent
            },
            'rate_limits': {
                'per_minute': {
                    'current': minute_count,
                    'max': self.max_per_minute,
                    'remaining': max(0, self.max_per_minute - minute_count)
                },
                'per_hour': {
                    'current': hour_count,
                    'max': self.max_per_hour,
                    'remaining': max(0, self.max_per_hour - hour_count)
                }
            },
            'config': {
                'queue_timeout': self.queue_timeout,
                'retry_delay': self.retry_delay
            }
        }
    
    def reset_counters(self):
        """重置计数器（主要用于测试）"""
        with self.history_lock:
            self.minute_requests.clear()
            self.hour_requests.clear()
        
        with self.concurrent_lock:
            self.current_concurrent = 0
        
        logger.info("LLM限流器计数器已重置")


class MockRateLimiter:
    """模拟限流器（当限流器禁用时使用）"""
    
    def __init__(self):
        self.enabled = False
    
    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """直接通过，不做任何限制"""
        yield
    
    def get_status(self) -> Dict[str, Any]:
        """返回禁用状态"""
        return {
            'enabled': False,
            'message': '限流器已禁用'
        }
    
    def reset_counters(self):
        """空操作"""
        pass


def create_rate_limiter(config: Dict[str, Any]) -> LLMRateLimiter:
    """工厂方法创建限流器"""
    if config.get('enabled', True):
        return LLMRateLimiter(config)
    else:
        return MockRateLimiter()
