# src/pipelines/spatial_localizer.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from src.backbones.dino_v2 import DINOv2Encoder

class SpatialLocalizer:
    def __init__(self, dino_encoder: DINOv2Encoder, config):
        self.encoder = dino_encoder
        self.config = config

    def find_candidates_in_regions(self, frames: list, query_vector: torch.Tensor, trois: list):
        """
        Tìm các bounding box ứng cử viên trong các TRoI đã cho.
        """
        print("\nStage 1.2: Spatial Localization within TRoIs...")
        candidate_boxes_by_frame = {}

        # Duyệt qua từng vùng thời gian quan tâm
        for start_frame, end_frame in trois:
            print(f"Scanning TRoI from frame {start_frame} to {end_frame}...")
            
            # Xử lý các frame trong vùng này theo batch
            for i in tqdm(range(start_frame, end_frame + 1, self.config.SPATIAL_BATCH_SIZE)):
                batch_indices = range(i, min(i + self.config.SPATIAL_BATCH_SIZE, end_frame + 1))
                if not batch_indices:
                    break
                
                batch_frames = [frames[j] for j in batch_indices]
                original_dims = [(f.width, f.height) for f in batch_frames]

                # 1. Lấy bản đồ đặc trưng 2D
                feature_maps = self.encoder.get_dense_feature_map(batch_frames) # (B, H', W', C)
                feature_maps = F.normalize(feature_maps, p=2, dim=-1)

                # 2. Tính heatmap
                # query_vector (1, C) -> (1, 1, 1, C) để broadcast
                heatmaps = F.cosine_similarity(query_vector.view(1, 1, 1, -1), feature_maps) # (B, H', W')

                # 3. Từ heatmap ra bounding box
                for j, heatmap in enumerate(heatmaps):
                    frame_idx = batch_indices[j]
                    w, h = original_dims[j]
                    
                    # Upscale heatmap về kích thước ảnh gốc
                    heatmap_np = heatmap.numpy()
                    heatmap_resized = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_CUBIC)
                    
                    # Áp dụng ngưỡng để tạo mask nhị phân
                    _, binary_mask = cv2.threshold(heatmap_resized, self.config.HEATMAP_THRESHOLD, 1, cv2.THRESH_BINARY)
                    binary_mask = (binary_mask * 255).astype(np.uint8)

                    # Tìm contours
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Chuyển contours thành bounding box
                    boxes = []
                    for contour in contours:
                        x, y, bw, bh = cv2.boundingRect(contour)
                        boxes.append([x, y, x + bw, y + bh])
                    
                    if boxes:
                        candidate_boxes_by_frame[frame_idx] = boxes
        
        print(f"Found candidate boxes in {len(candidate_boxes_by_frame)} frames.")
        return candidate_boxes_by_frame