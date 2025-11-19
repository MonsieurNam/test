# src/utils/video.py

import cv2
from PIL import Image
from tqdm import tqdm
import sys

def read_video_as_pil_images(video_path: str, max_frames: int = None):
    """
    Đọc một file video và trả về một danh sách các ảnh PIL.

    Args:
        video_path (str): Đường dẫn đến file video.
        max_frames (int, optional): Số lượng khung hình tối đa cần đọc. 
                                    Hữu ích cho việc debug nhanh. Mặc định là None (đọc hết).

    Returns:
        list[PIL.Image.Image]: Một danh sách các đối tượng ảnh PIL, mỗi ảnh là một khung hình.
    """
    # Kiểm tra xem video có tồn tại không
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return []
    except Exception as e:
        print(f"An error occurred while trying to open the video: {e}")
        return []

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Thiết lập thanh tiến trình (progress bar) với tqdm
    pbar_desc = f"Reading video frames from {video_path.split('/')[-1]}"
    with tqdm(total=frame_count, desc=pbar_desc, file=sys.stdout) as pbar:
        while True:
            ret, frame = cap.read()
            
            # Dừng lại nếu video kết thúc
            if not ret:
                break
            
            # OpenCV đọc ảnh theo định dạng BGR, cần chuyển sang RGB cho PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Chuyển đổi mảng numpy thành ảnh PIL
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            
            pbar.update(1)

            # Dừng lại nếu đã đạt số lượng frame tối đa (dùng cho debug)
            if max_frames and len(frames) >= max_frames:
                print(f"\nReached max_frames limit of {max_frames}.")
                break
    
    # Giải phóng tài nguyên
    cap.release()
    
    print(f"\nSuccessfully read {len(frames)} frames.")
    return frames