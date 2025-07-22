import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse
from pathlib import Path
import sys
import os

class SkeletonVisualizationDemo:
    """实时展示骨骼转换过程的演示程序"""
    
    def __init__(self, model_path='yolo11x-pose.pt'):
        """初始化演示程序"""
        self.model = YOLO(model_path)
        
        # YOLO到NTU的映射关系
        self.yolo_to_ntu_mapping = {
            0: 3,   # nose -> head
            5: 4,   # left shoulder -> left shoulder
            6: 8,   # right shoulder -> right shoulder
            7: 5,   # left elbow -> left elbow
            8: 9,   # right elbow -> right elbow
            9: 6,   # left wrist -> left wrist
            10: 10, # right wrist -> right wrist
            11: 12, # left hip -> left hip
            12: 16, # right hip -> right hip
            13: 13, # left knee -> left knee
            14: 17, # right knee -> right knee
            15: 14, # left ankle -> left ankle
            16: 18, # right ankle -> right ankle
        }
        
        # NTU骨骼连接关系 (用于绘制骨骼线条)
        self.ntu_connections = [
            (0, 1), (1, 2), (2, 3),  # 脊柱
            (2, 4), (4, 5), (5, 6), (6, 7),  # 左臂
            (2, 8), (8, 9), (9, 10), (10, 11),  # 右臂
            (0, 12), (12, 13), (13, 14), (14, 15),  # 左腿
            (0, 16), (16, 17), (17, 18), (18, 19),  # 右腿
            (1, 20),  # 脊柱中心
            (6, 21), (6, 22),  # 左手
            (10, 23), (10, 24),  # 右手
        ]
        
        # YOLO骨骼连接关系
        self.yolo_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上身
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 下身
            (5, 11), (6, 12),  # 身体连接
        ]
        
        # 颜色定义
        self.colors = {
            'yolo': (0, 255, 0),      # 绿色
            'ntu': (255, 0, 0),       # 红色
            'keypoint': (0, 255, 255), # 黄色
            'text': (255, 255, 255),   # 白色
            'background': (50, 50, 50) # 深灰色
        }
        
        # 窗口大小
        self.window_width = 1200
        self.window_height = 400
        
        print("骨骼转换演示程序初始化完成")
        print("按 'q' 退出程序")
    
    def yolo_to_ntu_conversion(self, yolo_keypoints):
        """将YOLO关键点转换为NTU格式"""
        # 初始化NTU关键点数组 (25个关键点)
        ntu_keypoints = np.zeros((25, 3), dtype=np.float32)
        
        # 直接映射
        for yolo_idx, ntu_idx in self.yolo_to_ntu_mapping.items():
            if yolo_idx < len(yolo_keypoints):
                ntu_keypoints[ntu_idx] = yolo_keypoints[yolo_idx]
        
        # 插值计算特殊关键点
        if len(yolo_keypoints) > 12:
            # 基础脊柱点 (髋部中心)
            if yolo_keypoints[11][2] > 0.3 and yolo_keypoints[12][2] > 0.3:
                ntu_keypoints[0] = (yolo_keypoints[11] + yolo_keypoints[12]) * 0.5
            
            # 颈部 (肩部中心)
            if yolo_keypoints[5][2] > 0.3 and yolo_keypoints[6][2] > 0.3:
                ntu_keypoints[2] = (yolo_keypoints[5] + yolo_keypoints[6]) * 0.5
            
            # 脊柱中部
            if ntu_keypoints[0][2] > 0.3 and ntu_keypoints[2][2] > 0.3:
                ntu_keypoints[1] = (ntu_keypoints[0] + ntu_keypoints[2]) * 0.5
            
            # 脊柱关节
            if ntu_keypoints[0][2] > 0.3 and ntu_keypoints[1][2] > 0.3:
                ntu_keypoints[20] = (ntu_keypoints[0] + ntu_keypoints[1]) * 0.5
        
        # 手部和脚部扩展
        self._extend_extremities(yolo_keypoints, ntu_keypoints)
        
        return ntu_keypoints
    
    def _extend_extremities(self, yolo_keypoints, ntu_keypoints):
        """扩展手部和脚部关键点"""
        # 左手扩展
        if len(yolo_keypoints) > 9 and yolo_keypoints[9][2] > 0.3:
            ntu_keypoints[7] = yolo_keypoints[9]  # 左手腕
            ntu_keypoints[21] = yolo_keypoints[9] + np.array([15, 0, 0])  # 左手指尖
            ntu_keypoints[22] = yolo_keypoints[9] + np.array([10, -5, 0])  # 左拇指
        
        # 右手扩展
        if len(yolo_keypoints) > 10 and yolo_keypoints[10][2] > 0.3:
            ntu_keypoints[11] = yolo_keypoints[10]  # 右手腕
            ntu_keypoints[23] = yolo_keypoints[10] + np.array([-15, 0, 0])  # 右手指尖
            ntu_keypoints[24] = yolo_keypoints[10] + np.array([-10, -5, 0])  # 右拇指
        
        # 脚部扩展
        if len(yolo_keypoints) > 15 and yolo_keypoints[15][2] > 0.3:
            ntu_keypoints[15] = yolo_keypoints[15]  # 左脚踝
        if len(yolo_keypoints) > 16 and yolo_keypoints[16][2] > 0.3:
            ntu_keypoints[19] = yolo_keypoints[16]  # 右脚踝
    
    def draw_skeleton(self, img, keypoints, connections, color, point_color=None, confidence_threshold=0.3, line_thickness=2, point_size=4):
        """绘制骨骼"""
        if point_color is None:
            point_color = color
        
        # 绘制连接线
        for connection in connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                # 检查置信度
                if len(pt1) > 2 and len(pt2) > 2:
                    if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                        cv2.line(img, (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), color, line_thickness)
        
        # 绘制关键点
        for i, point in enumerate(keypoints):
            if len(point) > 2 and point[2] > confidence_threshold:
                cv2.circle(img, (int(point[0]), int(point[1])), point_size, point_color, -1)
                cv2.putText(img, str(i), (int(point[0])+8, int(point[1])-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def scale_and_center_skeleton(self, keypoints, canvas_width, canvas_height, scale_factor=1.5):
        """缩放并居中骨骼关键点"""
        if keypoints is None or len(keypoints) == 0:
            return keypoints
        
        # 找到有效关键点
        valid_points = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_points) == 0:
            return keypoints
        
        # 计算边界框
        min_x, min_y = valid_points[:, 0].min(), valid_points[:, 1].min()
        max_x, max_y = valid_points[:, 0].max(), valid_points[:, 1].max()
        
        # 计算中心点和尺寸
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        width, height = max_x - min_x, max_y - min_y
        
        # 缩放关键点
        scaled_keypoints = keypoints.copy()
        for i in range(len(scaled_keypoints)):
            if scaled_keypoints[i, 2] > 0.3:  # 只处理有效点
                # 相对于中心点缩放
                scaled_keypoints[i, 0] = (scaled_keypoints[i, 0] - center_x) * scale_factor + canvas_width / 2
                scaled_keypoints[i, 1] = (scaled_keypoints[i, 1] - center_y) * scale_factor + canvas_height / 2
        
        return scaled_keypoints
    
    def create_display_frame(self, original_frame, yolo_keypoints, ntu_keypoints):
        """创建显示帧，包含三个视图"""
        # 调整原始帧大小
        frame_height, frame_width = original_frame.shape[:2]
        display_width = self.window_width // 3
        display_height = self.window_height
        
        # 创建显示画布
        display_frame = np.zeros((display_height, self.window_width, 3), dtype=np.uint8)
        display_frame[:] = self.colors['background']
        
        # 调整帧大小
        resized_frame = cv2.resize(original_frame, (display_width, display_height))
        
        # 1. 原始视频 (左侧)
        display_frame[:, :display_width] = resized_frame
        cv2.putText(display_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        # 2. YOLO检测结果 (中间) - 只显示骨骼
        yolo_canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        yolo_canvas[:] = self.colors['background']
        
        if yolo_keypoints is not None:
            # 调整关键点坐标到显示尺寸
            scaled_yolo = yolo_keypoints.copy()
            scaled_yolo[:, 0] = scaled_yolo[:, 0] * display_width / frame_width
            scaled_yolo[:, 1] = scaled_yolo[:, 1] * display_height / frame_height
            
            # 缩放并居中骨骼
            scaled_yolo = self.scale_and_center_skeleton(scaled_yolo, display_width, display_height, scale_factor=2.0)
            
            self.draw_skeleton(yolo_canvas, scaled_yolo, self.yolo_connections, 
                             self.colors['yolo'], self.colors['keypoint'], 
                             line_thickness=3, point_size=6)
            
            # 显示关键点数量
            cv2.putText(yolo_canvas, f"YOLO: {len(yolo_keypoints)} points", 
                       (10, display_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.colors['text'], 2)
        
        display_frame[:, display_width:display_width*2] = yolo_canvas
        cv2.putText(display_frame, "YOLO Detection", (display_width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        # 3. NTU转换结果 (右侧) - 只显示骨骼
        ntu_canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        ntu_canvas[:] = self.colors['background']
        
        if ntu_keypoints is not None:
            # 调整关键点坐标到显示尺寸
            scaled_ntu = ntu_keypoints.copy()
            scaled_ntu[:, 0] = scaled_ntu[:, 0] * display_width / frame_width
            scaled_ntu[:, 1] = scaled_ntu[:, 1] * display_height / frame_height
            
            # 缩放并居中骨骼
            scaled_ntu = self.scale_and_center_skeleton(scaled_ntu, display_width, display_height, scale_factor=2.0)
            
            self.draw_skeleton(ntu_canvas, scaled_ntu, self.ntu_connections, 
                             self.colors['ntu'], self.colors['keypoint'],
                             line_thickness=3, point_size=6)
            
            # 显示关键点数量
            valid_points = np.sum(ntu_keypoints[:, 2] > 0.3)
            cv2.putText(ntu_canvas, f"NTU: {valid_points}/25 points", 
                       (10, display_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.colors['text'], 2)
        
        display_frame[:, display_width*2:] = ntu_canvas
        cv2.putText(display_frame, "NTU Conversion", (display_width*2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        # 添加分割线
        cv2.line(display_frame, (display_width, 0), (display_width, display_height), 
                (255, 255, 255), 2)
        cv2.line(display_frame, (display_width*2, 0), (display_width*2, display_height), 
                (255, 255, 255), 2)
        
        # 添加说明文字
        cv2.putText(display_frame, "Press 'q' to quit, 's' to save screenshot", (10, display_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # 添加映射说明
        if yolo_keypoints is not None and ntu_keypoints is not None:
            cv2.putText(display_frame, "Green: YOLO -> Red: NTU", (self.window_width - 250, display_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return display_frame
    
    def run(self, source=0):
        """运行演示程序"""
        print(f"正在打开视频源: {source}")
        
        # 打开视频捕获
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频源: {source}")
            return
        
        # 设置视频属性
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("演示程序启动，正在处理视频流...")
        
        frame_count = 0
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break
            
            frame_count += 1
            
            try:
                # YOLO检测
                results = self.model(frame, verbose=False)
                yolo_keypoints = None
                ntu_keypoints = None
                
                if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                    yolo_keypoints = results[0].keypoints.data[0].cpu().numpy()
                    
                    # 转换为NTU格式
                    ntu_keypoints = self.yolo_to_ntu_conversion(yolo_keypoints)
                
                # 创建显示帧
                display_frame = self.create_display_frame(frame, yolo_keypoints, ntu_keypoints)
                
                # 计算FPS
                fps_counter += 1
                if time.time() - start_time >= 1.0:
                    fps = fps_counter / (time.time() - start_time)
                    fps_counter = 0
                    start_time = time.time()
                    
                    # 显示FPS
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                               (self.window_width - 100, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
                
                # 显示结果
                cv2.imshow('Skeleton Conversion Demo', display_frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存截图
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"skeleton_demo_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"截图已保存: {filename}")
                    
            except Exception as e:
                print(f"处理第 {frame_count} 帧时出错: {e}")
                continue
        
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("演示程序结束")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='骨骼转换演示程序')
    parser.add_argument('--source', type=str, default='0', 
                       help='视频源 (0=摄像头, 或视频文件路径)')
    parser.add_argument('--model', type=str, default='yolo11x-pose.pt',
                       help='YOLO模型路径')
    
    args = parser.parse_args()
    
    # 处理视频源参数
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        print("请确保YOLO模型文件在当前目录下")
        return
    
    # 创建并运行演示
    demo = SkeletonVisualizationDemo(args.model)
    demo.run(source)

if __name__ == "__main__":
    main()
