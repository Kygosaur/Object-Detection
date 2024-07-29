from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import torch

def get_incremented_path(base_path):
    """Get an incremented path if the file already exists."""
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_path)
    counter = 1
    while os.path.exists(f"{name}({counter}){ext}"):
        counter += 1
    return f"{name}({counter}){ext}"

def init_kalman_filters(num_keypoints):
    filters = []
    for _ in range(num_keypoints):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([0., 0., 0., 0.])  # initial state (x, y, dx, dy)
        kf.F = np.array([[1., 0., 1., 0.],
                        [0., 1., 0., 1.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])  # state transition matrix
        kf.H = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.]])  # measurement function
        kf.P *= 1000.  # covariance matrix
        kf.R = np.array([[1., 0.],
                        [0., 1.]])  # measurement noise
        kf.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)  # process noise
        filters.append(kf)
    return filters

def process_video(video_path: str, model: YOLO) -> None:
    """Process a video, extract keypoints, and save to Excel and video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Prepare output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_output.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    all_frames_data = []
    frame_count = 0
    num_keypoints = 17  # Number of keypoints in YOLO pose estimation
    kalman_filters = init_kalman_filters(num_keypoints)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        
        for person_idx, person_keypoints in enumerate(results[0].keypoints.data):
            frame_data = {
                'frame': frame_count,
                'person': person_idx,
            }
            
            keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            for i, keypoint in enumerate(person_keypoints):
                x, y, conf = keypoint
                
                kalman_filters[i].predict()
                kalman_filters[i].update(np.array([x.item(), y.item()]))
                
                filtered_x, filtered_y = kalman_filters[i].x[:2]
                
                frame_data[f'{keypoint_names[i]}_x'] = filtered_x
                frame_data[f'{keypoint_names[i]}_y'] = filtered_y
                frame_data[f'{keypoint_names[i]}_conf'] = conf.item()
            
            all_frames_data.append(frame_data)
        
        # Create a new tensor for visualization
        if results[0].keypoints.data.shape[0] > 0:
            filtered_keypoints = np.array([[kf.x[0], kf.x[1]] for kf in kalman_filters])
            filtered_keypoints_tensor = torch.tensor(filtered_keypoints, 
                                                    dtype=results[0].keypoints.data.dtype, 
                                                    device=results[0].keypoints.data.device)
            
            # Create a new keypoints tensor for visualization
            vis_keypoints = results[0].keypoints.data.clone()
            vis_keypoints[0, :, :2] = filtered_keypoints_tensor
            
            # Create a new Results object for visualization
            vis_results = results[0].clone()
            vis_results.keypoints.data = vis_keypoints
            
            annotated_frame = vis_results.plot()
        else:
            annotated_frame = results[0].plot()
        
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        frame_count += 1

    cap.release()
    out.release()

    # Create DataFrame and save to Excel
    df = pd.DataFrame(all_frames_data)
    output_excel_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_keypoints.xlsx'))
    df.to_excel(output_excel_path, index=False)
    
    print(f"Keypoint data saved to: {output_excel_path}")
    print(f"Processed video saved to: {output_video_path}")

def process_image(image_path: str, model: YOLO) -> None:
    """Process a single image, extract keypoints, and save to Excel and image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file {image_path}")
        return

    # Convert image to RGB (3 channels)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)
    
    all_data = []
    
    # Extract keypoints for each person
    for person_idx, person_keypoints in enumerate(results[0].keypoints.data):
        person_data = {
            'person': person_idx,
        }
        
        # YOLO keypoint order
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for i, keypoint in enumerate(person_keypoints):
            x, y, conf = keypoint
            person_data[f'{keypoint_names[i]}_x'] = x.item()
            person_data[f'{keypoint_names[i]}_y'] = y.item()
            person_data[f'{keypoint_names[i]}_conf'] = conf.item()
        
        all_data.append(person_data)
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(all_data)
    output_excel_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_keypoints.xlsx'))
    df.to_excel(output_excel_path, index=False)
    
    # Save annotated image
    annotated_image = results[0].plot()
    output_image_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_output.jpg'))
    cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    print(f"Keypoint data saved to: {output_excel_path}")
    print(f"Processed image saved to: {output_image_path}")

def main():
    model = YOLO('yolov8n-pose.pt')
    input_path = r"c:\Users\jack\Desktop\Untitled video - Made with Clipchamp.mp4"
    
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(input_path, model)
    else:
        process_video(input_path, model)

if __name__ == "__main__":
    main()