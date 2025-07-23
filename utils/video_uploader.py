"""
视频上传器 - 支持上传视频到OSS并获取URL
"""
import os
import time
import logging
from typing import Dict, Any, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class VideoUploader:
    """视频上传器，支持OSS云存储"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.provider = config.get('cloud_provider', 'oss')
        self.client = None
        self.bucket = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """初始化上传客户端"""
        if not self.enabled:
            logger.info("Video uploader is disabled")
            return True
            
        try:
            if self.provider == 'oss':
                return self._init_oss()
            else:
                logger.error(f"Unsupported cloud provider: {self.provider}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize video uploader: {e}")
            return False
    
    def _init_oss(self) -> bool:
        """初始化阿里云OSS"""
        try:
            import oss2
            
            oss_config = self.config.get('oss_config', {})
            access_key_id = oss_config.get('access_key_id')
            access_key_secret = oss_config.get('access_key_secret')
            endpoint = oss_config.get('endpoint')
            bucket_name = oss_config.get('bucket_name')
            
            if not all([access_key_id, access_key_secret, endpoint, bucket_name]):
                logger.error("OSS configuration incomplete")
                return False
            
            # 创建认证对象
            auth = oss2.Auth(access_key_id, access_key_secret)
            
            # 创建Bucket对象
            self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
            
            # 测试连接
            try:
                self.bucket.get_bucket_info()
                logger.info(f"OSS initialized successfully: {bucket_name}")
                self.is_initialized = True
                return True
            except Exception as e:
                logger.error(f"OSS connection test failed: {e}")
                return False
                
        except ImportError:
            logger.error("oss2 library not installed. Please install with: pip install oss2")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OSS: {e}")
            return False
    
    def upload_video(self, local_path: str, remote_key: Optional[str] = None) -> Optional[str]:
        """
        上传视频文件到OSS
        
        Args:
            local_path: 本地视频文件路径
            remote_key: OSS中的文件键名，如果为None则自动生成
            
        Returns:
            上传成功返回OSS URL，失败返回None
        """
        if not self.enabled:
            logger.debug("Video uploader is disabled, skipping upload")
            return None
            
        if not self.is_initialized:
            logger.error("Video uploader not initialized")
            return None
        
        if not os.path.exists(local_path):
            logger.error(f"Local video file not found: {local_path}")
            return None
        
        try:
            # 生成远程文件键名
            if remote_key is None:
                filename = os.path.basename(local_path)
                timestamp = int(time.time())
                remote_key = f"llm_videos/{timestamp}_{filename}"
            
            logger.info(f"Uploading video to OSS: {local_path} -> {remote_key}")
            
            # 上传文件
            start_time = time.time()
            result = self.bucket.put_object_from_file(remote_key, local_path)
            upload_time = time.time() - start_time
            
            if result.status == 200:
                # 构建OSS URL
                oss_config = self.config.get('oss_config', {})
                endpoint = oss_config.get('endpoint', '')
                bucket_name = oss_config.get('bucket_name', '')
                
                # 构建完整的OSS URL
                if endpoint.startswith('https://'):
                    base_url = endpoint.replace('https://', f'https://{bucket_name}.')
                elif endpoint.startswith('http://'):
                    base_url = endpoint.replace('http://', f'http://{bucket_name}.')
                else:
                    base_url = f"https://{bucket_name}.{endpoint}"
                
                oss_url = f"{base_url}/{remote_key}"
                
                logger.info(f"Video uploaded successfully in {upload_time:.2f}s: {oss_url}")
                return oss_url
            else:
                logger.error(f"Upload failed with status: {result.status}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upload video to OSS: {e}")
            return None
    
    def delete_video(self, remote_key: str) -> bool:
        """
        从OSS删除视频文件
        
        Args:
            remote_key: OSS中的文件键名
            
        Returns:
            删除成功返回True，失败返回False
        """
        if not self.enabled or not self.is_initialized:
            return False
        
        try:
            result = self.bucket.delete_object(remote_key)
            if result.status == 204:
                logger.info(f"Video deleted from OSS: {remote_key}")
                return True
            else:
                logger.error(f"Delete failed with status: {result.status}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete video from OSS: {e}")
            return False
    
    def get_video_url(self, remote_key: str, expires: int = 3600) -> Optional[str]:
        """
        获取视频的临时访问URL
        
        Args:
            remote_key: OSS中的文件键名
            expires: URL过期时间（秒）
            
        Returns:
            临时URL或None
        """
        if not self.enabled or not self.is_initialized:
            return None
        
        try:
            url = self.bucket.sign_url('GET', remote_key, expires)
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        self.client = None
        self.bucket = None
        self.is_initialized = False
        logger.info("Video uploader cleaned up")
    
    def get_status(self) -> Dict[str, Any]:
        """获取上传器状态"""
        return {
            'enabled': self.enabled,
            'provider': self.provider,
            'initialized': self.is_initialized
        }
