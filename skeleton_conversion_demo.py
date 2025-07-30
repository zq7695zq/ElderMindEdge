import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse
from pathlib import Path
import sys
import os

class SkeletonVisualizationDemo:
    """
    实时展示骨骼转换过程的演示程序。
    功能：
    - 实时显示YOLO骨骼、NTU骨骼转换、最终结果（裁剪后的人物）。
    - 可选地将处理结果逐帧保存。
    - 保存的纯骨骼图将自动裁剪到骨骼区域。
    """
    
    # ... __init__ 和其他未变动的方法保持原样 ...
    # 为了简洁，我将省略未变动的部分，只展示被修改和新增的核心代码。
    # 您只需将下面的代码替换掉之前版本中对应的部分即可。
    
    def __init__(self, model_path='yolov8n-pose.pt', save_output=False):
        """初始化演示程序"""
        self.model = YOLO(model_path)
        self.save_output = save_output
        self.output_dir = None
        self.original_dir = None
        self.overlaid_dir = None
        self.skeleton_only_dir = None
        self.current_action = "falling"
        self.current_confidence = 0.85
        self.yolo_to_ntu_mapping = {0: 3, 5: 4, 6: 8, 7: 5, 8: 9, 9: 6, 10: 10, 11: 12, 12: 16, 13: 13, 14: 17, 15: 14, 16: 18}
        self.ntu_connections = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (21, 22), (20, 8), (8, 9), (9, 10), (10, 11), (23, 24), (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]
        self.yolo_connections = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)]
        self.colors = {'yolo': (0, 255, 0), 'ntu': (255, 100, 0), 'keypoint': (0, 255, 255), 'text': (255, 255, 255), 'background': (40, 40, 40)}
        self.window_width, self.window_height = 1200, 400
        self.window_name = 'Skeleton Conversion Demo'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        print("骨骼转换演示程序初始化完成。按 'q' 退出。")

    # ==================== 核心修改：创建裁剪后的纯骨骼图 ====================
    def _create_cropped_skeleton_image(self, keypoints, connections, color):
        """
        为保存创建一个纯净的、仅包含骨骼并已裁剪到边界框的图像。
        """
        if keypoints is None:
            return None

        # 1. 找到所有有效关键点的边界
        valid_points = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_points) < 2:  # 至少需要两个点才能形成一个有意义的区域
            return None

        min_x, min_y = valid_points[:, :2].min(axis=0)
        max_x, max_y = valid_points[:, :2].max(axis=0)

        # 2. 增加一些边距，让骨骼不紧贴边缘
        margin = 40
        x1, y1 = int(min_x - margin), int(min_y - margin)
        x2, y2 = int(max_x + margin), int(max_y + margin)
        
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

        # 3. 创建一个仅够容纳裁剪区域的画布
        canvas = np.full((height, width, 3), self.colors['background'], dtype=np.uint8)

        # 4. 转换关键点坐标到新的（裁剪后的）画布坐标系
        translated_keypoints = keypoints.copy()
        # 只有当关键点有效时才进行平移
        valid_mask = translated_keypoints[:, 2] > 0.3
        translated_keypoints[valid_mask, 0] -= x1
        translated_keypoints[valid_mask, 1] -= y1
        
        # 5. 在新画布上绘制骨骼
        self.draw_skeleton(
            canvas, 
            translated_keypoints, 
            connections, 
            color, 
            point_color=self.colors['keypoint'],
            line_thickness=3,
            point_size=6
        )
        return canvas
    # ==================== 修改结束 ====================

    # ... (其他方法 yolo_to_ntu_conversion, draw_skeleton, _create_cropped_person_view, create_display_frame等保持不变)
    # 此处省略这些方法的代码，使用上一个回答中的版本即可。
    
    def yolo_to_ntu_conversion(self, yolo_keypoints):
        ntu_keypoints = np.zeros((25, 3), dtype=np.float32)
        for yolo_idx, ntu_idx in self.yolo_to_ntu_mapping.items():
            if yolo_idx < len(yolo_keypoints): ntu_keypoints[ntu_idx] = yolo_keypoints[yolo_idx]
        if len(yolo_keypoints) > 12:
            if yolo_keypoints[11][2]>0.3 and yolo_keypoints[12][2]>0.3: ntu_keypoints[0]=(yolo_keypoints[11]+yolo_keypoints[12])*0.5
            if yolo_keypoints[5][2]>0.3 and yolo_keypoints[6][2]>0.3: ntu_keypoints[2]=(yolo_keypoints[5]+yolo_keypoints[6])*0.5
            if ntu_keypoints[0][2]>0.3 and ntu_keypoints[2][2]>0.3: ntu_keypoints[1]=(ntu_keypoints[0]+ntu_keypoints[2])*0.5
            if ntu_keypoints[0][2]>0.3 and ntu_keypoints[1][2]>0.3: ntu_keypoints[20]=(ntu_keypoints[0]+ntu_keypoints[1])*0.5
        self._extend_extremities(yolo_keypoints, ntu_keypoints)
        return ntu_keypoints
    def _extend_extremities(self, yk, nk):
        if len(yk)>9 and yk[9][2]>0.3: nk[7]=yk[9]; nk[21]=yk[9]+np.array([15,0,0]); nk[22]=yk[9]+np.array([10,-5,0])
        if len(yk)>10 and yk[10][2]>0.3: nk[11]=yk[10]; nk[23]=yk[10]+np.array([-15,0,0]); nk[24]=yk[10]+np.array([-10,-5,0])
        if len(yk)>15 and yk[15][2]>0.3: nk[15]=yk[15]
        if len(yk)>16 and yk[16][2]>0.3: nk[19]=yk[16]
    def draw_skeleton(self, img, keypoints, connections, color, point_color=None, confidence_threshold=0.3, line_thickness=2, point_size=4, show_numbers=False):
        if point_color is None: point_color=color
        for pt1_idx, pt2_idx in connections:
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1, pt2 = keypoints[pt1_idx], keypoints[pt2_idx]
                if len(pt1)>2 and len(pt2)>2 and pt1[2]>confidence_threshold and pt2[2]>confidence_threshold: cv2.line(img,(int(pt1[0]),int(pt1[1])),(int(pt2[0]),int(pt2[1])),color,line_thickness,cv2.LINE_AA)
        for i, point in enumerate(keypoints):
            if len(point)>2 and point[2]>confidence_threshold:
                cv2.circle(img,(int(point[0]),int(point[1])),point_size,point_color,-1,cv2.LINE_AA)
                if show_numbers: cv2.putText(img,str(i),(int(point[0])+8,int(point[1])-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    def _create_cropped_person_view(self, original_frame, keypoints, target_w, target_h):
        canvas=np.full((target_h, target_w, 3),self.colors['background'],dtype=np.uint8)
        if keypoints is None: return canvas
        valid_points = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_points)<3: return canvas
        min_x,min_y=valid_points[:,:2].min(axis=0); max_x,max_y=valid_points[:,:2].max(axis=0)
        margin=30; x1,y1=max(0,int(min_x-margin)),max(0,int(min_y-margin)); x2,y2=min(original_frame.shape[1],int(max_x+margin)),min(original_frame.shape[0],int(max_y+margin))
        person_crop = original_frame[y1:y2, x1:x2]
        if person_crop.shape[0]==0 or person_crop.shape[1]==0: return canvas
        crop_h,crop_w=person_crop.shape[:2]; scale=min(target_w/crop_w,target_h/crop_h); new_w,new_h=int(crop_w*scale),int(crop_h*scale)
        resized_crop=cv2.resize(person_crop,(new_w,new_h),interpolation=cv2.INTER_AREA)
        paste_x,paste_y=(target_w-new_w)//2,(target_h-new_h)//2
        canvas[paste_y:paste_y+new_h,paste_x:paste_x+new_w]=resized_crop
        text=f"{self.current_action}: {self.current_confidence:.2f}"; (text_w,text_h),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
        cv2.rectangle(canvas,(5,5),(10+text_w,15+text_h),(0,0,0),-1); cv2.putText(canvas,text,(10,10+text_h),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
        return canvas
    def create_display_frame(self, original_frame, yolo_keypoints, ntu_keypoints=None):
        display_w_third = self.window_width//3; display_frame=np.zeros((self.window_height,self.window_width,3),dtype=np.uint8)
        left_view=original_frame.copy()
        if yolo_keypoints is not None: self.draw_skeleton(left_view,yolo_keypoints,self.yolo_connections,self.colors['yolo'],self.colors['keypoint'])
        display_frame[:,:display_w_third]=cv2.resize(left_view,(display_w_third,self.window_height))
        middle_view=np.full((self.window_height,display_w_third,3),self.colors['background'],dtype=np.uint8)
        cv2.putText(middle_view,"YOLO -> NTU Skeleton",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,self.colors['text'],2,cv2.LINE_AA)
        if ntu_keypoints is not None:
            scaled_ntu=self.scale_and_center_skeleton(ntu_keypoints,display_w_third,self.window_height)
            self.draw_skeleton(middle_view,scaled_ntu,self.ntu_connections,self.colors['ntu'],self.colors['keypoint'],line_thickness=3,point_size=5)
        display_frame[:,display_w_third:display_w_third*2]=middle_view
        right_view=self._create_cropped_person_view(original_frame,yolo_keypoints,display_w_third,self.window_height)
        display_frame[:,display_w_third*2:]=right_view
        return display_frame
    def scale_and_center_skeleton(self, keypoints, canvas_width, canvas_height, scale_factor=1.5):
        valid_points=keypoints[keypoints[:,2]>0.3]
        if len(valid_points)==0: return keypoints
        min_coords,max_coords=valid_points[:,:2].min(axis=0),valid_points[:,:2].max(axis=0)
        center,skel_height=(min_coords+max_coords)/2,max_coords[1]-min_coords[1]
        if skel_height==0: return keypoints
        scale=(canvas_height*0.7)/skel_height*scale_factor
        scaled_keypoints=keypoints.copy()
        for i in range(len(scaled_keypoints)):
            if scaled_keypoints[i,2]>0.3:
                scaled_keypoints[i,0]=(scaled_keypoints[i,0]-center[0])*scale+canvas_width/2
                scaled_keypoints[i,1]=(scaled_keypoints[i,1]-center[1])*scale+canvas_height/2
        return scaled_keypoints


    def run(self, source=0):
        """运行演示程序"""
        print(f"正在打开视频源: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频源: {source}"); return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        if self.save_output:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"output_{timestamp}")
            self.original_dir = self.output_dir / "original_frames"
            self.overlaid_dir = self.output_dir / "overlaid_frames"
            self.skeleton_only_dir = self.output_dir / "skeleton_only_frames"
            self.original_dir.mkdir(parents=True, exist_ok=True)
            self.overlaid_dir.mkdir(parents=True, exist_ok=True)
            self.skeleton_only_dir.mkdir(parents=True, exist_ok=True)
            print(f"输出将被保存到: {self.output_dir.resolve()}")
        
        print("演示程序启动，正在处理视频流...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束。"); break
            
            frame_count += 1
            try:
                results = self.model(frame, verbose=False)
                yolo_keypoints, ntu_keypoints = None, None
                if results[0].keypoints and len(results[0].keypoints.data) > 0:
                    yolo_keypoints = results[0].keypoints.data[0].cpu().numpy()
                    ntu_keypoints = self.yolo_to_ntu_conversion(yolo_keypoints)
                
                display_frame = self.create_display_frame(frame, yolo_keypoints, ntu_keypoints)
                cv2.imshow(self.window_name, display_frame)

                # --- 保存帧的逻辑 ---
                if self.save_output and yolo_keypoints is not None:
                    frame_filename = f"frame_{frame_count:06d}.png"

                    # 1. 保存原始图像
                    cv2.imwrite(str(self.original_dir / frame_filename), frame)

                    # 2. 保存原始图像+骨骼图 (全尺寸)
                    overlaid_frame = frame.copy()
                    self.draw_skeleton(overlaid_frame, yolo_keypoints, self.yolo_connections, self.colors['yolo'], self.colors['keypoint'], line_thickness=2, point_size=5)
                    cv2.imwrite(str(self.overlaid_dir / frame_filename), overlaid_frame)

                    # ==================== 核心修改：调用新的裁剪函数 ====================
                    # 3. 创建并保存裁剪后的纯骨骼图
                    skeleton_only_img = self._create_cropped_skeleton_image(
                        yolo_keypoints, self.yolo_connections, self.colors['yolo']
                    )
                    
                    # 只有当成功创建图像时才保存
                    if skeleton_only_img is not None:
                        cv2.imwrite(str(self.skeleton_only_dir / frame_filename), skeleton_only_img)
                    # ==================== 修改结束 ====================

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"live_display_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"实时显示窗口截图已保存: {filename}")
            
            except Exception as e:
                import traceback
                print(f"处理第 {frame_count} 帧时出错: {e}")
                traceback.print_exc()
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        print("演示程序结束")

# main 函数无需任何改动
def main():
    parser = argparse.ArgumentParser(description='骨骼转换与可视化演示程序')
    parser.add_argument('--source', type=str, default='0', help='视频源 (0=摄像头, 或视频文件路径)')
    parser.add_argument('--model', type=str, default='yolo11m-pose.pt', help='YOLOv8姿态估计模型路径 (例如yolo11m-pose.pt)')
    parser.add_argument('--save-output', action='store_true', help='如果设置此标志，则将处理结果逐帧保存到新目录中')
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在于 '{args.model}'")
        return
    demo = SkeletonVisualizationDemo(model_path=args.model, save_output=args.save_output)
    demo.run(source=source)

if __name__ == "__main__":
    main()