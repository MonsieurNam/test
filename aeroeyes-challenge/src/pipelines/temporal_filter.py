# src/pipelines/temporal_filter.py

import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.video import read_video_as_pil_images
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
        print("Stage 1.1: Temporal Filtering...")
        
        # 1. Tạo Query Vector
        query_vector = self._compute_query_vector(ref_images_pil)

        # 2. Đọc video và tạo chuỗi tín hiệu
        frames = read_video_as_pil_images(video_path)
        num_frames = len(frames)
        similarity_scores = np.zeros(num_frames)

        print(f"Processing {num_frames} frames to generate temporal signal...")
        for i in tqdm(range(0, num_frames, self.config.TEMPORAL_BATCH_SIZE)):
            batch_frames = frames[i:i+self.config.TEMPORAL_BATCH_SIZE]
            if not batch_frames:
                break
            
            frame_embeddings = self.encoder.get_cls_embedding(batch_frames)
            frame_embeddings = F.normalize(frame_embeddings, p=2, dim=1)
            
            # Tính độ tương đồng
            scores = F.cosine_similarity(query_vector, frame_embeddings)
            similarity_scores[i:i+len(batch_frames)] = scores.numpy()

        # 3. Lọc và tìm đỉnh
        smoothed_scores = gaussian_filter1d(similarity_scores, sigma=self.config.SIGNAL_SMOOTHING_SIGMA)
        peaks, _ = find_peaks(smoothed_scores, prominence=self.config.PEAK_PROMINENCE)

        if len(peaks) == 0:
            print("No significant peaks found. The object might not be in the video.")
            return []

        # 4. Tạo các Vùng Thời gian Quan tâm (TRoI)
        trois = []
        for peak in peaks:
            start_frame = max(0, peak - self.config.ROI_WINDOW_FRAMES)
            end_frame = min(num_frames - 1, peak + self.config.ROI_WINDOW_FRAMES)
            trois.append((start_frame, end_frame))
        
        # Gộp các vùng bị chồng lấn
        # (Bạn có thể thêm logic gộp phức tạp hơn nếu cần)
        print(f"Found {len(trois)} Temporal Region(s) of Interest.")
        return trois, frames, query_vector