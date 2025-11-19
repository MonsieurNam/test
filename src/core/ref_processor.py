import torch
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from torchvision import transforms

class ReferenceProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"[INFO] Đang tải mô hình DINOv2 (ViT-L/14) về {self.device}...")
        
        # Load DINOv2 từ Torch Hub (Tự động tải weight SOTA)
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.dinov2.to(self.device)
        self.dinov2.eval() # Chế độ đánh giá, không train
        
        # Chuẩn hóa ảnh theo chuẩn ImageNet (Yêu cầu của DINOv2)
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)), # 518x518 là kích thước tối ưu cho DINOv2
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[INFO] Reference Processor đã sẵn sàng!")

    def remove_background(self, image_path):
        """
        Đọc ảnh và xóa nền bằng rembg
        """
        # Đọc ảnh bằng PIL
        img = Image.open(image_path).convert("RGB")
        
        # Xóa nền (trả về ảnh RGBA)
        img_no_bg = remove(img)
        
        # Chuyển về nền xám trung tính (thay vì trong suốt) để đưa vào DINOv2
        # Tạo ảnh nền xám
        background = Image.new('RGB', img_no_bg.size, (128, 128, 128))
        # Dán ảnh vật thể lên nền xám dựa trên mask alpha
        background.paste(img_no_bg, mask=img_no_bg.split()[3])
        
        return background, img_no_bg

    def extract_features(self, pil_image):
        """
        Đưa ảnh PIL qua DINOv2 để lấy vector đặc trưng
        """
        # Tiền xử lý
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            # DINOv2 trả về dictionary, ta lấy output chính
            features = self.dinov2(img_tensor)
            
        # Features có shape [1, 1024], ta chuyển về vector 1 chiều và chuẩn hóa
        features = features[0]
        features = torch.nn.functional.normalize(features, dim=0)
        
        return features.cpu().numpy()

    def process_reference_images(self, image_paths):
        """
        Hàm chính: Nhận list đường dẫn ảnh -> Trả về Feature Bank
        """
        feature_bank = []
        processed_images = [] # Để visualize kiểm tra
        
        for path in image_paths:
            # 1. Xóa nền
            clean_img, viz_img = self.remove_background(path)
            processed_images.append(viz_img)
            
            # 2. Trích xuất đặc trưng
            embedding = self.extract_features(clean_img)
            feature_bank.append(embedding)
            
        # Tính vector trung bình (Mean Embedding)
        feature_bank = np.array(feature_bank)
        mean_embedding = np.mean(feature_bank, axis=0)
        
        # Chuẩn hóa lại vector trung bình
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        
        return mean_embedding, feature_bank, processed_images