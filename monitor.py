import cv2
import os
import pandas as pd
import numpy as np
import time
import psutil
import GPUtil
import threading
from ultralytics import YOLO

class SystemMonitor:
    def __init__(self):
        self.cpu_percent = 0
        self.ram_percent = 0
        self.gpu_util = 0
        self.gpu_mem = 0
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._monitor, daemon=True).start()

    def stop(self):
        self.running = False

    def _monitor(self):
        while self.running:
            self.cpu_percent = psutil.cpu_percent()
            self.ram_percent = psutil.virtual_memory().percent
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assume we're using the first GPU
                self.gpu_util = gpu.load * 100
                self.gpu_mem = gpu.memoryUsed
            time.sleep(1)  # Update every second

    def get_metrics(self):
        return {
            'cpu': self.cpu_percent,
            'ram': self.ram_percent,
            'gpu_util': self.gpu_util,
            'gpu_mem': self.gpu_mem
        }

class PoseEstimator:
    def __init__(self, model_path='yolov8l-pose.pt'):
        self.model = YOLO(model_path)

    def process_frame(self, frame, frame_count):
        start_time = time.time()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        
        frame_data = self._extract_keypoints(results, frame_count)
        annotated_frame = results[0].plot()
        
        end_time = time.time()
        process_time = end_time - start_time
        fps = 1 / process_time
        
        return frame_data, annotated_frame, fps

    def _extract_keypoints(self, results, frame_count):
        all_data = []
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for person_idx, person_keypoints in enumerate(results[0].keypoints.data):
            person_data = {'person': person_idx, 'frame': frame_count}
            for i, keypoint in enumerate(person_keypoints):
                x, y, conf = keypoint
                person_data[f'{keypoint_names[i]}_x'] = x.item()
                person_data[f'{keypoint_names[i]}_y'] = y.item()
                person_data[f'{keypoint_names[i]}_conf'] = conf.item()
            all_data.append(person_data)
        
        return all_data

class VideoProcessor:
    def __init__(self, input_path, output_path, pose_estimator, system_monitor):
        self.input_path = input_path
        self.output_path = output_path
        self.pose_estimator = pose_estimator
        self.system_monitor = system_monitor
        self.all_frames_data = []
        self.fps_data = []
        self.system_data = []

    def process(self):
        cap = cv2.VideoCapture(self.input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.input_path}")
            return

        out = self._setup_output_video(cap)

        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_data, annotated_frame, fps = self.pose_estimator.process_frame(frame, frame_count)
            self.all_frames_data.extend(frame_data)
            self.fps_data.append({'frame': frame_count, 'fps': fps})
            
            system_metrics = self.system_monitor.get_metrics()
            system_metrics['frame'] = frame_count
            self.system_data.append(system_metrics)
            
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            frame_count += 1

        cap.release()
        out.release()

    def _setup_output_video(self, cap):
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

    def save_data(self, excel_path):
        keypoints_df = pd.DataFrame(self.all_frames_data)
        fps_df = pd.DataFrame(self.fps_data)
        system_df = pd.DataFrame(self.system_data)
        
        avg_fps = sum(d['fps'] for d in self.fps_data) / len(self.fps_data)
        avg_cpu = sum(d['cpu'] for d in self.system_data) / len(self.system_data)
        avg_ram = sum(d['ram'] for d in self.system_data) / len(self.system_data)
        avg_gpu_util = sum(d['gpu_util'] for d in self.system_data) / len(self.system_data)
        avg_gpu_mem = sum(d['gpu_mem'] for d in self.system_data) / len(self.system_data)

        averages_df = pd.DataFrame({
            'Metric': ['Average FPS', 'Average CPU Usage (%)', 'Average RAM Usage (%)', 'Average GPU Utilization (%)', 'Average GPU Memory Usage (MB)'],
            'Value': [avg_fps, avg_cpu, avg_ram, avg_gpu_util, avg_gpu_mem]
        })

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            keypoints_df.to_excel(writer, sheet_name='Keypoints', index=False)
            fps_df.to_excel(writer, sheet_name='FPS', index=False)
            system_df.to_excel(writer, sheet_name='System Metrics', index=False)
            averages_df.to_excel(writer, sheet_name='Averages', index=False)

def get_incremented_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_path)
    counter = 1
    while os.path.exists(f"{name}({counter}){ext}"):
        counter += 1
    return f"{name}({counter}){ext}"

def main():
    input_path = r"c:\Users\jack\Desktop\Untitled video - Made with Clipchamp.mp4"
    output_video_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_output.mp4'))
    output_excel_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_data.xlsx'))

    pose_estimator = PoseEstimator()
    system_monitor = SystemMonitor()
    system_monitor.start()

    video_processor = VideoProcessor(input_path, output_video_path, pose_estimator, system_monitor)
    video_processor.process()
    video_processor.save_data(output_excel_path)

    system_monitor.stop()

    print(f"Processed video saved to: {output_video_path}")
    print(f"Data saved to: {output_excel_path}")

if __name__ == "__main__":
    main()