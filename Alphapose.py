import cv2
import torch
from alphapose.utils.config import update_config
from alphapose.models import builder
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.writer import DataWriter

class AlphaPoseProcessor:
    def __init__(self, cfg_path, checkpoint_path):
        self.cfg = update_config(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize pose model
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.pose_model.to(self.device)
        self.pose_model.eval()

        # Initialize detector
        self.detector = DetectionLoader(self.cfg, self.device)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        return self._process_frame(img)

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self._process_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()

    def _process_frame(self, frame):
        # Prepare input
        inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(inp).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(self.device)

        # Get detections
        with torch.no_grad():
            detections = self.detector.detect(inp, self.cfg.DATASET.INPUT_SIZE)

        # Get pose estimation
        with torch.no_grad():
            hm = self.pose_model(inp)
            # You may need to apply additional post-processing here based on AlphaPose's specific requirements

        # Visualize results
        # This is a placeholder - you'll need to implement proper visualization based on AlphaPose's output format
        for detection in detections:
            x1, y1, x2, y2, _ = detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        return frame

if __name__ == "__main__":
    cfg_path = "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
    checkpoint_path = "pretrained_models/halpe26_fast_res50_256x192.pth"
    
    processor = AlphaPoseProcessor(cfg_path, checkpoint_path)

    # Process an image
    processor.process_image("path/to/your/image.jpg")

    # Process a video
    processor.process_video(r"c:\Users\jack\Desktop\Untitled video - Made with Clipchamp.mp4")