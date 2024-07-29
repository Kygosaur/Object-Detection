import mediapipe as mp
import cv2
import os

def initialize_mediapipe(static_image_mode=True):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=static_image_mode, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_drawing, mp_pose, pose

def process_image(image_path, mp_drawing, mp_pose, pose):
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return None

    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(image)

    cap.release()
    out.release()
    return 'output_video.mp4'

def save_output(output, filename):
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    output_path = os.path.join(desktop, filename)
    
    if isinstance(output, str):  # If output is a video file path
        os.rename(output, output_path)
    else:  # If output is an image
        cv2.imwrite(output_path, output)
    
    print(f"Output saved to: {output_path}")

def is_image_file(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def main():
    input_path = r"c:\Users\jack\Desktop\Untitled video - Made with Clipchamp.mp4"  # to input vid or img
    
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

if __name__ == "__main__":
    main()