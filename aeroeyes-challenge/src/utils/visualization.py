# src/utils/visualization.py

import cv2
import numpy as np
from typing import List

def draw_boxes_on_frame(
    frame_np: np.ndarray, 
    boxes: List[List[int]], 
    color: tuple = (0, 255, 0), 
    thickness: int = 2,
    labels: List[str] = None
) -> np.ndarray:
    """
    Vẽ một hoặc nhiều bounding box lên một khung hình.

    Args:
        frame_np (np.ndarray): Khung hình dưới dạng mảng numpy (định dạng BGR của OpenCV).
        boxes (List[List[int]]): Danh sách các bounding box. 
                                  Mỗi box có định dạng [x1, y1, x2, y2].
        color (tuple, optional): Màu của box theo định dạng BGR. Mặc định là xanh lá.
        thickness (int, optional): Độ dày của đường viền box. Mặc định là 2.
        labels (List[str], optional): Danh sách các nhãn tương ứng với mỗi box.

    Returns:
        np.ndarray: Khung hình đã được vẽ thêm các bounding box.
    """
    # Tạo một bản sao của frame để không làm thay đổi frame gốc
    output_frame = frame_np.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box) # Đảm bảo tọa độ là số nguyên
        
        # Vẽ hình chữ nhật
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)

        # Nếu có nhãn, vẽ thêm nhãn
        if labels and i < len(labels):
            label = labels[i]
            
            # Lấy kích thước của text để vẽ nền
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Vẽ một hình chữ nhật nền cho text
            cv2.rectangle(output_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1) # -1 là fill
            
            # Viết text lên trên nền
            cv2.putText(output_frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Text màu đen

    return output_frame

def draw_heatmap_on_frame(
    frame_np: np.ndarray,
    heatmap_np: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Chồng một heatmap lên trên một khung hình.

    Args:
        frame_np (np.ndarray): Khung hình gốc (BGR).
        heatmap_np (np.ndarray): Heatmap 2D, giá trị từ 0 đến 1.
        alpha (float, optional): Độ trong suốt của heatmap. Mặc định là 0.5.

    Returns:
        np.ndarray: Khung hình đã được chồng heatmap.
    """
    if heatmap_np.shape[:2] != frame_np.shape[:2]:
        heatmap_np = cv2.resize(heatmap_np, (frame_np.shape[1], frame_np.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Chuyển heatmap (0-1) thành heatmap màu (0-255, 3 kênh)
    heatmap_colored = cv2.applyColorMap((heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Trộn ảnh gốc và heatmap
    overlayed_frame = cv2.addWeighted(frame_np, 1 - alpha, heatmap_colored, alpha, 0)

    return overlayed_frame