# main_stage1.py

import os
from PIL import Image
import cv2
import numpy as np

from src.backbones.dino_v2 import DINOv2Encoder
from src.pipelines.temporal_filter import TemporalFilter
from src.pipelines.spatial_localizer import SpatialLocalizer
from src.utils.visualization import draw_boxes_on_frame
from configs import stage1_config as config

def main(video_id: str):
    # --- 1. SETUP ---
    video_path = os.path.join(config.DATASET_DIR, video_id, "drone_video.mp4")
    ref_images_dir = os.path.join(config.DATASET_DIR, video_id, "object_images")
    
    output_video_path = os.path.join(config.OUTPUT_DIR, f"{video_id}_stage1_output.mp4")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    ref_images_pil = [
        Image.open(os.path.join(ref_images_dir, fname)).convert("RGB")
        for fname in os.listdir(ref_images_dir)
    ]

    # --- 2. INITIALIZE MODELS AND PIPELINES ---
    dino_encoder = DINOv2Encoder(model_name=config.DINO_MODEL_NAME)
    temporal_filter = TemporalFilter(dino_encoder, config)
    spatial_localizer = SpatialLocalizer(dino_encoder, config)

    # --- 3. RUN STAGE 1 PIPELINES ---
    # --- SỬA LỖI ---
    # Đảm bảo chỉ nhận 2 giá trị trả về: trois và query_vector
    trois, query_vector = temporal_filter.find_regions_of_interest(video_path, ref_images_pil)
    
    if not trois:
        print("No objects to localize. Exiting.")
        return

    # Truyền video_path thay vì list frames
    candidate_boxes = spatial_localizer.find_candidates_in_regions(video_path, query_vector, trois)

    # --- 4. VISUALIZATION ---
    print(f"\nVisualizing results and saving to {output_video_path}...")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in candidate_boxes:
            frame = draw_boxes_on_frame(frame, candidate_boxes[frame_idx], color=(0, 255, 0), thickness=2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Visualization complete.")


if __name__ == '__main__':
    target_video_id = "Jacket_0" 
    main(target_video_id)