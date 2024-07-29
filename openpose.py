import cv2
import numpy as np
from openpose import pyopenpose as op
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def name_keypoints(keypoints):
    keypoint_names = [
        "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
        "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip", "Right Knee",
        "Right Ankle", "Left Hip", "Left Knee", "Left Ankle", "Right Eye",
        "Left Eye", "Right Ear", "Left Ear"
    ]
    
    named_keypoints = []
    for person in keypoints:
        person_keypoints = {name: keypoint.tolist() for name, keypoint in zip(keypoint_names, person)}
        named_keypoints.append(person_keypoints)
    
    return named_keypoints

def process_frame(opWrapper, frame):
    datum = op.Datum()
    datum.cvInputData = frame
    try:
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        return datum
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return None

def main(args):
    # Configure OpenPose
    params = {
        "model_folder": args.model_folder,
        "number_people_max": args.max_people,
        "disable_blending": not args.enable_blending  # Disable blending for faster processing
    }

    # Start OpenPose
    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
    except Exception as e:
        logger.error(f"Error initializing OpenPose: {e}")
        return

    if args.use_video:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            logger.error("Error opening video file")
            return
    else:
        frame = cv2.imread(args.input)
        if frame is None:
            logger.error("Error reading image file")
            return

    frame_count = 0
    start_time = time.time()

    while True:
        if args.use_video:
            ret, frame = cap.read()
            if not ret:
                break
        
        datum = process_frame(opWrapper, frame)
        if datum is None:
            continue

        frame_count += 1

        keypoints = datum.poseKeypoints
        if keypoints is not None and keypoints.size > 0:
            named_keypoints = name_keypoints(keypoints)
            
            for i, person in enumerate(named_keypoints):
                logger.debug(f"Person {i + 1}:")
                for name, keypoint in person.items():
                    logger.debug(f"  {name}: {keypoint}")

        if args.display:
            cv2.imshow("OpenPose Result", datum.cvOutputData)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not args.use_video:
            break

    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    logger.info(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds ({fps:.2f} FPS)")

    if args.use_video:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenPose Demo")
    parser.add_argument("--input", required=True, help="Path to input image or video file")
    parser.add_argument("--use_video", action="store_true", help="Process video instead of image")
    parser.add_argument("--model_folder", default="../models/", help="Path to OpenPose models")
    parser.add_argument("--max_people", type=int, default=1, help="Maximum number of people to detect")
    parser.add_argument("--enable_blending", action="store_true", help="Enable blending of OpenPose output")
    parser.add_argument("--display", action="store_true", help="Display output")
    args = parser.parse_args()

    main(args)