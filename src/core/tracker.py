import os
import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam2_video_predictor import build_sam2_video_predictor

class Sam2Tracker:
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        print("[INFO] Đang khởi tạo SAM 2 Tracker...")
        self.device = device
        # Khởi tạo mô hình
        self.predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        self.inference_state = None
        self.video_frames_dir = None

    def prepare_video(self, video_path, output_dir="temp_frames"):
        """
        SAM 2 cần input là folder ảnh JPEG. Hàm này giải nén video ra ảnh.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Xóa ảnh cũ nếu có để tránh lẫn lộn
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

        print(f"[INFO] Đang trích xuất frames từ video: {video_path}...")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        frame_names = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Lưu tên file dạng 00000.jpg, 00001.jpg...
            file_name = f"{frame_idx:05d}.jpg"
            save_path = os.path.join(output_dir, file_name)
            cv2.imwrite(save_path, frame)
            frame_names.append(file_name)
            frame_idx += 1
            
        cap.release()
        self.video_frames_dir = output_dir
        print(f"[INFO] Đã trích xuất {frame_idx} frames vào {output_dir}")
        
        # Khởi tạo state cho SAM 2
        self.inference_state = self.predictor.init_state(video_path=output_dir)

    def add_anchor(self, frame_idx, points, labels):
        """
        Thêm gợi ý (point prompts) vào frame cụ thể
        points: List các tọa độ [[x, y], ...]
        labels: List nhãn [1, ...] (1 là foreground, 0 là background)
        """
        if self.inference_state is None:
            raise ValueError("Chưa chạy prepare_video!")

        print(f"[INFO] Thêm Anchor tại Frame {frame_idx}: Points {points}")
        
        # SAM 2 yêu cầu dtype là float32
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        # Gọi hàm add_new_points_or_box (API mới của SAM 2)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=1, # ID của vật thể (chúng ta chỉ tìm 1 vật thể nên để là 1)
            points=points,
            labels=labels,
        )
        return out_mask_logits

    def propagate(self):
        """
        Lan truyền mask ra toàn bộ video
        """
        print("[INFO] Bắt đầu lan truyền (Tracking)...")
        results = {} # {frame_idx: mask_binary}
        
        # propagate_in_video trả về generator
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            # out_mask_logits shape: [N_obj, H, W] -> Chọn object 0
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze() # Binary mask
            
            # Chỉ lưu nếu mask có pixel (có vật thể)
            if mask.sum() > 0:
                results[out_frame_idx] = mask
                
        print(f"[INFO] Hoàn tất! Tìm thấy vật thể trong {len(results)} frames.")
        return results