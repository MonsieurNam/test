# src/pipelines/temporal_filter.py

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2 # Thêm import cv2 để lấy num_frames

from src.utils.video import frame_generator
from src.backbones.dino_v2 import DINOv2Encoder

class TemporalFilter:
    def __init__(self, dino_encoder: DINOv2Encoder, config):
        self.encoder = dino_encoder
        self.config = config

    def _compute_query_vector(self, ref_images_pil):
        """Tính vector truy vấn từ ảnh tham chiếu."""
        ref_embeddings = self.encoder.get_cls_embedding(ref_images_pil)
        query_vector = F.normalize(ref_embeddings.mean(dim=0), p=2, dim=0)
        return query_vector.unsqueeze(0) # Thêm chiều batch

    def find_regions_of_interest(self, video_path: str, ref_images_pil: list):
        """
        Chạy toàn bộ pipeline lọc thời gian để tìm ra các TRoI.
        """
        print("Stage 1.1: Temporal Filtering (Streaming Mode)...")
        
        query_vector = self._compute_query_vector(ref_images_pil)

        # Lấy tổng số frame để hiển thị progress bar
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if num_frames == 0:
            print("Warning: Video has no frames or could not be read.")
            return [], query_vector

        similarity_scores = np.zeros(num_frames)
        
        video_gen = frame_generator(video_path)
        
        batch_frames = []
        frame_idx = 0
        
        with tqdm(total=num_frames, desc="Generating temporal signal") as pbar:
            for frame_pil in video_gen:
                batch_frames.append(frame_pil)
                
                if len(batch_frames) == self.config.TEMPORAL_BATCH_SIZE or frame_idx == num_frames - 1:
                    if not batch_frames:
                        break
                    
                    frame_embeddings = self.encoder.get_cls_embedding(batch_frames)
                    frame_embeddings = F.normalize(frame_embeddings, p=2, dim=1)
                    
                    scores = F.cosine_similarity(query_vector, frame_embeddings)
                    
                    start_idx = frame_idx - len(batch_frames) + 1
                    end_idx = frame_idx + 1
                    similarity_scores[start_idx:end_idx] = scores.numpy()
                    
                    pbar.update(len(batch_frames))
                    batch_frames = []
                
                frame_idx += 1

        smoothed_scores = gaussian_filter1d(similarity_scores, sigma=self.config.SIGNAL_SMOOTHING_SIGMA)
        peaks, _ = find_peaks(smoothed_scores, prominence=self.config.PEAK_PROMINENCE)

        if len(peaks) == 0:
            print("No significant peaks found.")
            # --- SỬA LỖI ---
            # Trả về một tuple rỗng và query_vector để đảm bảo nhất quán
            return [], query_vector 

        trois = []
        for peak in peaks:
            start_frame = max(0, peak - self.config.ROI_WINDOW_FRAMES)
            end_frame = min(num_frames - 1, peak + self.config.ROI_WINDOW_FRAMES)
            trois.append((start_frame, end_frame))
        
        # (Tùy chọn) Thêm logic gộp các TRoI bị chồng lấn nếu cần
        
        print(f"Found {len(trois)} Temporal Region(s) of Interest.")
        
        return trois, query_vector