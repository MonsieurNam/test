# src/pipelines/spatial_localizer.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.backbones.dino_v2 import DINOv2Encoder

class SpatialLocalizer:
    """
    Lớp này chịu trách nhiệm định vị các ứng cử viên không gian (bounding boxes)
    bên trong các Vùng Thời gian Quan tâm (TRoIs) đã được xác định trước.
    Nó hoạt động ở chế độ streaming để tiết kiệm bộ nhớ.
    """
    def __init__(self, dino_encoder: DINOv2Encoder, config):
        """
        Khởi tạo SpatialLocalizer.

        Args:
            dino_encoder (DINOv2Encoder): Một instance của bộ mã hóa DINOv2.
            config: Module cấu hình chứa các tham số như SPATIAL_BATCH_SIZE.
        """
        self.encoder = dino_encoder
        self.config = config

    def find_candidates_in_regions(self, video_path: str, query_vector: torch.Tensor, trois: List[Tuple[int, int]]) -> Dict[int, List[List[int]]]:
        """
        Tìm các bounding box ứng cử viên trong các TRoI đã cho.

        Args:
            video_path (str): Đường dẫn đến file video.
            query_vector (torch.Tensor): Vector truy vấn đã được tính toán từ ảnh tham chiếu.
            trois (List[Tuple[int, int]]): Danh sách các TRoI, mỗi TRoI là một tuple (start_frame, end_frame).

        Returns:
            Dict[int, List[List[int]]]: Một dictionary, với key là chỉ số frame và
                                        value là danh sách các bounding box ứng cử viên
                                        có định dạng [x1, y1, x2, y2].
        """
        print("\nStage 1.2: Spatial Localization within TRoIs (Streaming Mode)...")
        candidate_boxes_by_frame = {}
        
        # Mở luồng video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video for spatial localization: {video_path}")
            return {}

        try:
            # Duyệt qua từng vùng thời gian quan tâm
            for start_frame, end_frame in trois:
                print(f"Scanning TRoI from frame {start_frame} to {end_frame}...")
                
                # Đặt con trỏ video đến frame bắt đầu của TRoI
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Đọc trước tất cả các frame cần thiết trong TRoI này vào RAM.
                # Vì TRoI thường ngắn, điều này sẽ không gây tràn RAM và nhanh hơn
                # so với việc đọc từng frame một.
                frames_in_roi = []
                for _ in range(start_frame, end_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Chuyển đổi sang PIL.Image để tương thích với DINOv2Encoder
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_in_roi.append(Image.fromarray(frame_rgb))

                # Xử lý các frame trong vùng này theo batch
                pbar_desc = f"TRoI [{start_frame}-{end_frame}]"
                for i in tqdm(range(0, len(frames_in_roi), self.config.SPATIAL_BATCH_SIZE), desc=pbar_desc):
                    batch_frames_pil = frames_in_roi[i : i + self.config.SPATIAL_BATCH_SIZE]
                    if not batch_frames_pil:
                        break
                    
                    # Lấy chỉ số frame toàn cục tương ứng với batch hiện tại
                    current_global_indices = range(start_frame + i, start_frame + i + len(batch_frames_pil))
                    original_dims = [(f.width, f.height) for f in batch_frames_pil]

                    # 1. Lấy bản đồ đặc trưng 2D
                    feature_maps = self.encoder.get_dense_feature_map(batch_frames_pil) # (B, H', W', C)
                    feature_maps = F.normalize(feature_maps, p=2, dim=-1)

                    # 2. Tính heatmap bằng cách so sánh với query vector
                    # query_vector (1, C) -> (1, 1, 1, C) để broadcast
                    heatmaps = F.cosine_similarity(query_vector.view(1, 1, 1, -1), feature_maps) # (B, H', W')

                    # 3. Từ heatmap ra bounding box cho từng ảnh trong batch
                    for j, heatmap in enumerate(heatmaps):
                        frame_idx = current_global_indices[j]
                        w, h = original_dims[j]
                        
                        heatmap_np = heatmap.numpy()
                        boxes = self._heatmap_to_bounding_boxes(heatmap_np, (w, h))
                        
                        if boxes:
                            candidate_boxes_by_frame[frame_idx] = boxes
        finally:
            # Đảm bảo luồng video luôn được đóng
            cap.release()
            
        print(f"Found candidate boxes in {len(candidate_boxes_by_frame)} frames.")
        return candidate_boxes_by_frame

    def _heatmap_to_bounding_boxes(self, heatmap_np: np.ndarray, original_shape: Tuple[int, int]) -> List[List[int]]:
        """
        Chuyển đổi một heatmap 2D thành một danh sách các bounding box.

        Args:
            heatmap_np (np.ndarray): Heatmap 2D (H', W') với giá trị từ -1 đến 1.
            original_shape (Tuple[int, int]): Kích thước ảnh gốc (width, height).

        Returns:
            List[List[int]]: Danh sách các bounding box [x1, y1, x2, y2].
        """
        w, h = original_shape
        
        # Upscale heatmap về kích thước ảnh gốc để có vị trí chính xác
        heatmap_resized = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Chuẩn hóa lại heatmap về khoảng 0-255 để xử lý ảnh
        # Chuyển từ [-1, 1] sang [0, 1] rồi nhân 255
        heatmap_normalized = ((heatmap_resized + 1) / 2 * 255).astype(np.uint8)

        # Áp dụng ngưỡng để tạo mask nhị phân
        # Ngưỡng HEATMAP_THRESHOLD cũng cần được chuyển đổi sang thang 0-255
        threshold_value = (self.config.HEATMAP_THRESHOLD + 1) / 2 * 255
        _, binary_mask = cv2.threshold(heatmap_normalized, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Sử dụng các phép toán hình thái để làm mịn mask và loại bỏ nhiễu nhỏ
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Tìm các vùng liên thông (contours) trên mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chuyển mỗi contour thành một bounding box
        boxes = []
        for contour in contours:
            # Bỏ qua các contour quá nhỏ
            if cv2.contourArea(contour) < 50: # Ngưỡng diện tích tối thiểu
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            boxes.append([x, y, x + bw, y + bh])
        
        return boxes