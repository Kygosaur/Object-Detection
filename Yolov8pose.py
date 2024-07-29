from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np

def get_incremented_path(base_path):
    """Get an incremented path if the file already exists."""
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_path)
    counter = 1
    while os.path.exists(f"{name}({counter}){ext}"):
        counter += 1
    return f"{name}({counter}){ext}"

def setup_output_video(cap, output_path):
    """Set up the output video writer."""
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def extract_keypoints(results):
    """Extract keypoints from YOLO results."""
    all_data = []
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    for person_idx, person_keypoints in enumerate(results[0].keypoints.data):
        person_data = {'person': person_idx}
        for i, keypoint in enumerate(person_keypoints):
            x, y, conf = keypoint
            person_data[f'{keypoint_names[i]}_x'] = x.item()
            person_data[f'{keypoint_names[i]}_y'] = y.item()
            person_data[f'{keypoint_names[i]}_conf'] = conf.item()
        all_data.append(person_data)
    
    return all_data

def save_to_excel(data, output_path):
    """Save data to Excel file."""
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print(f"Keypoint data saved to: {output_path}")

def process_frame(model, frame, frame_count):
    """Process a single frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    
    frame_data = extract_keypoints(results)
    for data in frame_data:
        data['frame'] = frame_count
    
    annotated_frame = results[0].plot()
    return frame_data, annotated_frame

def process_video(video_path: str, model: YOLO) -> None:
    """Process a video, extract keypoints, and save to Excel and video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    output_video_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_output.mp4'))
    out = setup_output_video(cap, output_video_path)

    all_frames_data = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_data, annotated_frame = process_frame(model, frame, frame_count)
        all_frames_data.extend(frame_data)
        
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        frame_count += 1

    cap.release()
    out.release()

    output_excel_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_keypoints.xlsx'))
    save_to_excel(all_frames_data, output_excel_path)
    
    print(f"Processed video saved to: {output_video_path}")

def process_image(image_path: str, model: YOLO) -> None:
    """Process a single image, extract keypoints, and save to Excel and image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file {image_path}")
        return

    all_data, annotated_image = process_frame(model, image, 0)
    
    output_excel_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_keypoints.xlsx'))
    save_to_excel(all_data, output_excel_path)
    
    output_image_path = get_incremented_path(os.path.join(os.path.expanduser('~'), 'Desktop', 'pose_estimation_output.jpg'))
    cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    print(f"Processed image saved to: {output_image_path}")

def process_file(file_path: str, model: YOLO) -> None:
    """Process either an image or video file."""
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(file_path, model)
    else:
        process_video(file_path, model)

def main():
    model = YOLO('yolov8l-pose.pt')
    input_path = r"c:\Users\jack\Desktop\Untitled video - Made with Clipchamp.mp4"
    process_file(input_path, model)

if __name__ == "__main__":
    main()