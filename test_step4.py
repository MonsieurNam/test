import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dataloader import AeroEyesDataset
from src.core.ref_processor import ReferenceProcessor
from src.core.scout import DinoScout
from src.core.tracker import Sam2Tracker

def main():
    # --- CẤU HÌNH ---
    SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
    SAM2_CONFIG = "sam2_hiera_l.yaml"
    SAMPLE_IDX = 0 # Backpack_0
    TARGET_FRAME = 3483 # Frame mà Scout đã tìm thấy balo
    
    # 1. KHỞI ĐỘNG HỆ THỐNG
    dataset = AeroEyesDataset("dataset/")
    sample = dataset.get_sample(SAMPLE_IDX)
    print(f"=== TEST TRACKING: {sample['video_id']} ===")
    
    # 2. CHẠY SCOUT (Lại bước cũ để lấy tọa độ)
    print("--- Giai đoạn 1: Scout (Tìm vị trí) ---")
    ref_processor = ReferenceProcessor(device='cuda')
    ref_vector, _, _ = ref_processor.process_reference_images(sample['ref_images_paths'])
    scout = DinoScout(model=ref_processor.dinov2, device='cuda')
    
    cap = cv2.VideoCapture(sample['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Quét với thông số tối ưu từ bước trước
    detections, _ = scout.scan_frame(frame_rgb, ref_vector, crop_size=160, overlap=0.6)
    
    if not detections:
        print("Scout không tìm thấy gì! Dừng.")
        return
        
    best_det = detections[0] # Lấy cái tốt nhất
    bbox = best_det['bbox']
    score = best_det['score']
    
    # Tính tâm của Box -> Đây là Point Prompt cho SAM 2
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    print(f"-> Scout tìm thấy tại: {bbox} (Score {score:.3f})")
    print(f"-> Anchor Point: ({center_x}, {center_y})")

    # 3. CHẠY TRACKER (SAM 2)
    print("\n--- Giai đoạn 2: Tracker (Lan truyền) ---")
    tracker = Sam2Tracker(SAM2_CHECKPOINT, SAM2_CONFIG, device='cuda')
    
    # Giải nén video (Mất chút thời gian)
    tracker.prepare_video(sample['video_path'])
    
    # Thêm Anchor Point
    # labels=[1] nghĩa là điểm này thuộc về vật thể (positive)
    tracker.add_anchor(frame_idx=TARGET_FRAME, points=[[center_x, center_y]], labels=[1])
    
    # Lan truyền!
    masks = tracker.propagate()
    
    # 4. VISUALIZE KẾT QUẢ (Tạo video output)
    print("\n--- Giai đoạn 3: Xuất Video Kết quả ---")
    h, w = frame.shape[:2]
    out_path = "tracking_result.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Quay lại đầu video
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Nếu frame này có mask
        if frame_idx in masks:
            mask = masks[frame_idx]
            
            # Vẽ mask màu đỏ lên frame
            # Tạo overlay màu đỏ
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 2] = 255 # Kênh R
            
            # Áp dụng mask
            mask_indices = mask > 0
            frame[mask_indices] = cv2.addWeighted(frame[mask_indices], 0.5, colored_mask[mask_indices], 0.5, 0)
            
            # Vẽ thêm viền bounding box từ mask
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Backpack", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0: print(f"Đang ghi frame {frame_idx}...", end='\r')
        
    cap.release()
    out.release()
    print(f"\n[XONG] Video kết quả đã lưu tại: {out_path}")
    print("Hãy tải video về và kiểm tra xem SAM 2 có bám theo cái balo không!")

if __name__ == "__main__":
    main()