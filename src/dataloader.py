import os
import json
import cv2
import matplotlib.pyplot as plt
from glob import glob

class AeroEyesDataset:
    def __init__(self, root_dir, split="train"):
        """
        root_dir: Đường dẫn đến thư mục dataset (chứa thư mục 'samples' và 'annotations')
        """
        self.root_dir = root_dir
        self.samples_dir = os.path.join(root_dir, "samples")
        self.annot_path = os.path.join(root_dir, "annotations", "annotations.json")
        
        # Load annotations
        with open(self.annot_path, 'r') as f:
            self.annotations = json.load(f)
            
        # Tạo mapping từ video_id sang annotation để tra cứu nhanh
        self.annot_map = {item['video_id']: item for item in self.annotations}
        
        # Lấy danh sách tất cả các folder video
        self.video_folders = sorted(glob(os.path.join(self.samples_dir, "*")))
        
        print(f"[INFO] Đã load {len(self.video_folders)} mẫu video từ {self.root_dir}")

    def __len__(self):
        return len(self.video_folders)

    def get_sample(self, idx):
        """
        Lấy thông tin của một mẫu dữ liệu theo index
        """
        video_folder_path = self.video_folders[idx]
        video_id = os.path.basename(video_folder_path) # vd: drone_video_001
        
        # Đường dẫn video
        video_path = os.path.join(video_folder_path, "drone_video.mp4")
        
        # Đường dẫn 3 ảnh tham chiếu
        ref_images_paths = sorted(glob(os.path.join(video_folder_path, "object_images", "*.jpg")))
        
        # Lấy ground truth (nếu có)
        gt_data = self.annot_map.get(video_id, {}).get('annotations', [])
        
        return {
            "video_id": video_id,
            "video_path": video_path,
            "ref_images_paths": ref_images_paths,
            "ground_truth": gt_data
        }

    def visualize_sample(self, idx):
        sample = self.get_sample(idx)
        print(f"--- Visualizing: {sample['video_id']} ---")
        
        # Load 3 ảnh tham chiếu
        ref_imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in sample['ref_images_paths']]
        
        # Load video
        cap = cv2.VideoCapture(sample['video_path'])
        
        # --- CHỈNH SỬA Ở ĐÂY: Chúng ta sẽ nhảy đến frame có chứa vật thể đầu tiên ---
        start_frame = 0
        if sample['ground_truth']:
            # Lấy frame đầu tiên mà vật thể xuất hiện trong annotation
            start_frame = sample['ground_truth'][0]['bboxes'][0]['frame']
            
        # Set video đến đúng frame đó
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print("Lỗi: Không đọc được video.")
            return

        # Vẽ Ground Truth (CHỈ VẼ BBOX CỦA FRAME HIỆN TẠI)
        if sample['ground_truth']:
            for annot in sample['ground_truth']:
                for bbox in annot['bboxes']:
                    # Chỉ vẽ nếu frame của bbox trùng với frame mình đang xem (start_frame)
                    if bbox['frame'] == start_frame:
                        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Màu xanh lá
                        cv2.putText(frame, f"GT Frame: {bbox['frame']}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Plotting
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for i, img in enumerate(ref_imgs):
            axs[i].imshow(img)
            axs[i].set_title(f"Ref {i+1}")
            axs[i].axis('off')
            
        axs[3].imshow(frame)
        axs[3].set_title(f"Video at Frame {start_frame}")
        axs[3].axis('off')
        
        plt.show()

# Test code chạy trực tiếp
if __name__ == "__main__":
    # Đổi đường dẫn này thành đường dẫn thực tế trên máy bạn
    DATASET_PATH = "dataset/" 
    
    if os.path.exists(DATASET_PATH):
        dataset = AeroEyesDataset(DATASET_PATH)
        # Thử visualize mẫu đầu tiên
        dataset.visualize_sample(0)
    else:
        print(f"Không tìm thấy thư mục dataset tại: {DATASET_PATH}")
        print("Hãy đảm bảo bạn đã giải nén dữ liệu đúng cấu trúc.")