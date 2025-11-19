import cv2
import torch
import matplotlib.pyplot as plt
from src.dataloader import AeroEyesDataset
from src.core.ref_processor import ReferenceProcessor
from src.core.scout import DinoScout

def main():
    dataset = AeroEyesDataset("dataset/")
    sample = dataset.get_sample(0) # Backpack_0
    
    print(f"--- DEBUGGING LOCAL MATCHING: {sample['video_id']} ---")
    
    ref_processor = ReferenceProcessor(device='cuda')
    ref_vector, _, _ = ref_processor.process_reference_images(sample['ref_images_paths'])
    
    scout = DinoScout(model=ref_processor.dinov2, device='cuda')
    
    # Frame có vật thể
    target_frame_idx = 3483
    if sample['ground_truth']:
        target_frame_idx = sample['ground_truth'][0]['bboxes'][0]['frame']

    cap = cv2.VideoCapture(sample['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- QUÉT ĐA KÍCH THƯỚC (MULTI-SCALE) ---
    # Thử các crop size khác nhau để tìm cái nào khớp nhất
    scales = [160, 256] 
    all_detections = []
    all_patches = [] # Lưu trữ tạm để visualize (chỉ lưu top patches sau này)
    
    for crop_s in scales:
        print(f"-> Quét với crop_size={crop_s}...")
        dets, patches = scout.scan_frame(frame_rgb, ref_vector, crop_size=crop_s, overlap=0.6)
        
        # Cập nhật lại index patch để vẽ đúng
        # (Lưu patch vào một list riêng cho scale này thì hơi phức tạp để vẽ chung)
        # Ở đây ta vẽ riêng từng scale
        
        print(f"   Top 1 Score (Size {crop_s}): {dets[0]['score']:.4f}")
        
        # Vẽ Top 3 của Scale này
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Top Matches - Crop Size {crop_s}")
        for i in range(min(3, len(dets))):
            det = dets[i]
            plt.subplot(1, 3, i+1)
            plt.imshow(patches[det['patch_idx']])
            plt.title(f"Score: {det['score']:.3f}")
            plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()