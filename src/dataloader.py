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
        """
        Hàm debug: Hiển thị 3 ảnh tham chiếu và frame đầu tiên của video
        """
        sample = self.get_sample(idx)
        print(f"--- Visualizing: {sample['video_id']} ---")
        
        # Load 3 ảnh tham chiếu
        ref_imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in sample['ref_images_paths']]
        
        # Load frame đầu tiên của video
        cap = cv2.VideoCapture(sample['video_path'])
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print("Lỗi: Không đọc được video.")
            return

        # Vẽ Ground Truth lên frame (nếu có) - Lấy GT ở frame đầu tiên tìm thấy
        gt_bboxes = []
        if sample['ground_truth']:
            # Tìm annotation đầu tiên có dữ liệu
            first_annot = sample['ground_truth'][0] 
            if 'bboxes' in first_annot:
                for bbox in first_annot['bboxes']:
                    # Chỉ vẽ nếu frame trùng khớp (để demo thì vẽ đại diện)
                    # Ở đây vẽ ví dụ frame bất kỳ trong GT để check tọa độ
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, f"Frame: {bbox['frame']}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        # Plotting
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        # Hiển thị 3 ảnh object
        for i, img in enumerate(ref_imgs):
            axs[i].imshow(img)
            axs[i].set_title(f"Ref Image {i+1}")
            axs[i].axis('off')
            
        # Hiển thị frame video
        axs[3].imshow(frame)
        axs[3].set_title(f"Video Frame (w/ GT sample)")
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