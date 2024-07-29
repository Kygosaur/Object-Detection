import mediapipe as mp
import cv2
import os
import pandas as pd
import numpy as np
from typing import Optional, Union

def initialize_mediapipe(static_image_mode=True):
    """Initialize MediaPipe pose estimation."""
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=static_image_mode, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_drawing, mp_pose, pose

def calculate_movement_score(prev_landmarks, current_landmarks):
    """Calculate the movement score between two sets of landmarks."""
    if prev_landmarks is None or current_landmarks is None:
        return 0
    
    movement = sum(np.sqrt((prev.x - curr.x)**2 + (prev.y - curr.y)**2) 
                   for prev, curr in zip(prev_landmarks.landmark, current_landmarks.landmark))
    return movement / len(current_landmarks.landmark)

def process_image(image_path, mp_drawing, mp_pose, pose):
    """Process a single image and draw pose landmarks."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return image

def process_video(video_path, mp_drawing, mp_pose, pose):
    """Process a video, draw pose landmarks, and save keypoint data."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    prev_landmarks = None
    movement_threshold = 0.01  # Adjust this value based on your needs

    keypoint_data = []

    # Landmark names for reference
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_heel", "right_heel", "left_foot_index", "right_foot_index"
    ]

    frame_count = 0
    while True:
        success, image = cap.read()
        if not success:
            break

        frame_count += 1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            movement_score = calculate_movement_score(prev_landmarks, results.pose_landmarks)
            
            status = "Moving" if movement_score > movement_threshold else "Still"
            cv2.putText(image, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status == "Moving" else (0, 0, 255), 2)
            
            prev_landmarks = results.pose_landmarks

            frame_data = {'frame': frame_count}
            for idx, (landmark, name) in enumerate(zip(results.pose_landmarks.landmark, landmark_names)):
                frame_data[f'{name}_x'] = landmark.x
                frame_data[f'{name}_y'] = landmark.y
                frame_data[f'{name}_z'] = landmark.z
                frame_data[f'{name}_visibility'] = landmark.visibility
            keypoint_data.append(frame_data)

        out.write(image)

    cap.release()
    out.release()

    df = pd.DataFrame(keypoint_data)

    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    excel_path = os.path.join(desktop, 'keypoint_data.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Keypoint data saved to: {excel_path}")

    return 'output_video.mp4'

def save_output(output, filename):
    """Save the output (video or image) to the desktop."""
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_path = os.path.join(desktop, filename)
    
    if isinstance(output, str):  # If output is a video file path
        os.rename(output, output_path)
    else:  # If output is an image
        cv2.imwrite(output_path, output)
    
    print(f"Output saved to: {output_path}")

def is_image_file(file_path):
    """Check if the file is an image based on its extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def main():
    input_path = r"c:\Users\jack\Desktop\Untitled video - Made with Clipchamp.mp4"
    
    print(f"Processing file: {input_path}")
    
    try:
        if is_image_file(input_path):
            mp_drawing, mp_pose, pose = initialize_mediapipe(static_image_mode=True)
            output = process_image(input_path, mp_drawing, mp_pose, pose)
            if output is not None:
                save_output(output, "pose_estimation_output.jpg")
            else:
                print("Failed to process the image.")
        else:
            mp_drawing, mp_pose, pose = initialize_mediapipe(static_image_mode=False)
            output = process_video(input_path, mp_drawing, mp_pose, pose)
            if output is not None:
                save_output(output, "pose_estimation_output.mp4")
            else:
                print("Failed to process the video.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()