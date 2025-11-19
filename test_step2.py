import os
import matplotlib.pyplot as plt
import numpy as np
from src.dataloader import AeroEyesDataset
from src.core.ref_processor import ReferenceProcessor

def main():
    # 1. Load dataset
    dataset = AeroEyesDataset("dataset/")
    
    # 2. Khởi tạo Processor (Tự động tải DINOv2 lần đầu, sẽ mất chút thời gian)
    # Nếu máy không có GPU, đổi device='cpu'
    processor = ReferenceProcessor(device='cuda')
    
    # 3. Lấy mẫu đầu tiên để test (Ví dụ: Cái Balo)
    sample_idx = 0 
    sample = dataset.get_sample(sample_idx)
    print(f"\n--- Đang xử lý mẫu: {sample['video_id']} ---")
    print(f"Ảnh gốc: {sample['ref_images_paths']}")
    
    # 4. Chạy xử lý
    mean_emb, bank, clean_imgs = processor.process_reference_images(sample['ref_images_paths'])
    
    print(f"\n[KẾT QUẢ]:")
    print(f"Feature Bank Shape: {bank.shape} (3 ảnh x 1024 chiều)")
    print(f"Mean Embedding Shape: {mean_emb.shape} (Vector đại diện)")
    print(f"Độ lớn vector (Check chuẩn hóa): {np.linalg.norm(mean_emb):.4f}")
    
    # 5. Visualize kết quả Xóa nền
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Kết quả Xóa Nền & Feature Extraction - {sample['video_id']}", fontsize=16)
    
    for i, img in enumerate(clean_imgs):
        axs[i].imshow(img)
        axs[i].set_title(f"Ref {i+1} (No BG)")
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()