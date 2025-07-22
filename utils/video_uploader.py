"""
视频上传器 - 支持多种云存储服务
"""
import os
import time
import logging
from typing import Dict, Any, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class VideoUploader:
    """视频上传器，支持多种云存储"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('cloud_provider', 'oss')
        self.client = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """初始化上传客户端"""
        try:
            if self.provider == 'oss':
                return self._init_oss()
            elif self.provider == 'cos':
                return self._init_cos()
            elif self.provider == 's3':
                return self._init_s3()
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
            
            auth = oss2.Auth(access_key_id, access_key_secret)
            self.client = oss2.Bucket(auth, endpoint, bucket_name)
            
            self.is_initialized = True
            logger.info("OSS uploader initialized successfully")
            return True
            
        except ImportError:
            logger.error("oss2 library not installed. Please install with: pip install oss2")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OSS: {e}")
            return False
    
    def _init_cos(self) -> bool:
        """初始化腾讯云COS"""
        try:
            from qcloud_cos import CosConfig, CosS3Client
            
            cos_config = self.config.get('cos_config', {})
            secret_id = cos_config.get('secret_id')
            secret_key = cos_config.get('secret_key')
            region = cos_config.get('region')
            bucket_name = cos_config.get('bucket_name')
            
            if not all([secret_id, secret_key, region, bucket_name]):
                logger.error("COS configuration incomplete")
                return False
            
            config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
            self.client = CosS3Client(config)
            self.bucket_name = bucket_name
            
            self.is_initialized = True
            logger.info("COS uploader initialized successfully")
            return True
            
        except ImportError:
            logger.error("cos-python-sdk-v5 library not installed. Please install with: pip install cos-python-sdk-v5")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize COS: {e}")
            return False
    
    def _init_s3(self) -> bool:
        """初始化AWS S3"""
        try:
            import boto3
            
            s3_config = self.config.get('s3_config', {})
            access_key_id = s3_config.get('access_key_id')
            secret_access_key = s3_config.get('secret_access_key')
            region = s3_config.get('region')
            bucket_name = s3_config.get('bucket_name')
            
            if not all([access_key_id, secret_access_key, region, bucket_name]):
                logger.error("S3 configuration incomplete")
                return False
            
            self.client = boto3.client(
                's3',
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region
            )
            self.bucket_name = bucket_name
            
            self.is_initialized = True
            logger.info("S3 uploader initialized successfully")
            return True
            
        except ImportError:
            logger.error("boto3 library not installed. Please install with: pip install boto3")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize S3: {e}")
            return False
    
    def upload_video(self, video_path: str) -> Dict[str, Any]:
        """上传视频文件"""
        if not self.is_initialized:
            return {'success': False, 'error': 'Uploader not initialized'}
        
        if not os.path.exists(video_path):
            return {'success': False, 'error': f'Video file not found: {video_path}'}
        
        try:
            # 生成对象键
            filename = os.path.basename(video_path)
            timestamp = int(time.time())
            object_key = f"videos/{timestamp}_{filename}"
            
            if self.provider == 'oss':
                return self._upload_to_oss(video_path, object_key)
            elif self.provider == 'cos':
                return self._upload_to_cos(video_path, object_key)
            elif self.provider == 's3':
                return self._upload_to_s3(video_path, object_key)
            else:
                return {'success': False, 'error': f'Unsupported provider: {self.provider}'}
                
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            return {'success': False, 'error': str(e)}
    
    def _upload_to_oss(self, video_path: str, object_key: str) -> Dict[str, Any]:
        """上传到阿里云OSS"""
        try:
            result = self.client.put_object_from_file(object_key, video_path)
            
            # 构建URL
            oss_config = self.config.get('oss_config', {})
            endpoint = oss_config.get('endpoint')
            bucket_name = oss_config.get('bucket_name')
            
            if endpoint.startswith('http'):
                base_url = endpoint.replace('://', f'://{bucket_name}.')
            else:
                base_url = f"https://{bucket_name}.{endpoint}"
            
            url = urljoin(base_url + '/', object_key)
            
            return {
                'success': True,
                'url': url,
                'object_key': object_key,
                'etag': result.etag
            }
            
        except Exception as e:
            logger.error(f"OSS upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _upload_to_cos(self, video_path: str, object_key: str) -> Dict[str, Any]:
        """上传到腾讯云COS"""
        try:
            with open(video_path, 'rb') as fp:
                response = self.client.put_object(
                    Bucket=self.bucket_name,
                    Body=fp,
                    Key=object_key
                )
            
            # 构建URL
            cos_config = self.config.get('cos_config', {})
            region = cos_config.get('region')
            url = f"https://{self.bucket_name}.cos.{region}.myqcloud.com/{object_key}"
            
            return {
                'success': True,
                'url': url,
                'object_key': object_key,
                'etag': response.get('ETag', '')
            }
            
        except Exception as e:
            logger.error(f"COS upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _upload_to_s3(self, video_path: str, object_key: str) -> Dict[str, Any]:
        """上传到AWS S3"""
        try:
            with open(video_path, 'rb') as fp:
                self.client.upload_fileobj(fp, self.bucket_name, object_key)
            
            # 构建URL
            s3_config = self.config.get('s3_config', {})
            region = s3_config.get('region')
            url = f"https://{self.bucket_name}.s3.{region}.amazonaws.com/{object_key}"
            
            return {
                'success': True,
                'url': url,
                'object_key': object_key
            }
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def cleanup(self):
        """清理资源"""
        self.client = None
        self.is_initialized = False
