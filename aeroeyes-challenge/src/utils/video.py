# src/utils/video.py

import cv2
from PIL import Image
from tqdm import tqdm
import sys

def read_video_as_pil_images(video_path: str, max_frames: int = None):
    """
    CẢNH BÁO: HÀM NÀY SẼ LOAD TOÀN BỘ VIDEO VÀO RAM.
    Chỉ sử dụng cho các video rất ngắn hoặc khi debug.
    """
    # ... (giữ nguyên code cũ) ...
    pass

def frame_generator(video_path: str):
    """
    Tạo một generator để đọc từng frame của video một cách hiệu quả về bộ nhớ.

    Args:
        video_path (str): Đường dẫn đến file video.

    Yields:
        PIL.Image.Image: Một đối tượng ảnh PIL cho mỗi khung hình.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame_rgb)
    finally:
        # Đảm bảo tài nguyên luôn được giải phóng
        cap.release()