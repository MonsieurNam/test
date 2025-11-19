import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

class DinoScout:
    def __init__(self, model=None, device='cuda'):
        self.device = device
        
        if model:
            self.dinov2 = model
        else:
            print(f"[INFO] Scout đang tải lại DINOv2...")
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.dinov2.to(self.device)
            self.dinov2.eval()

        self.transform = transforms.Compose([
            transforms.Resize((518, 518)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print("[INFO] Scout (Local Feature Matching) đã sẵn sàng!")

    def get_sliding_windows(self, image, crop_size=224, overlap=0.5):
        h, w = image.shape[:2]
        stride = int(crop_size * (1 - overlap))
        patches_pil = []
        coords = [] 
        img_pil = Image.fromarray(image)
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x1 = x
                y1 = y
                x2 = min(x + crop_size, w)
                y2 = min(y + crop_size, h)
                if (x2 - x1) < crop_size: x1 = max(0, w - crop_size); x2 = w
                if (y2 - y1) < crop_size: y1 = max(0, h - crop_size); y2 = h
                
                patch = img_pil.crop((x1, y1, x2, y2))
                patches_pil.append(patch)
                coords.append((x1, y1, x2, y2))
                if x2 == w: break
            if y2 == h: break
        return patches_pil, coords

    def scan_frame(self, frame, ref_vector, crop_size=224, overlap=0.5):
        # 1. Cắt patches
        patches_pil, coords = self.get_sliding_windows(frame, crop_size=crop_size, overlap=overlap)
        if not patches_pil: return [], []

        # 2. Preprocess
        batch_tensors = [self.transform(p) for p in patches_pil]
        input_batch = torch.stack(batch_tensors).to(self.device)
        
        # 3. Inference lấy PATCH TOKENS (Local Features)
        with torch.no_grad():
            # forward_features trả về dict, ta lấy 'x_norm_patchtokens'
            # Shape: [Batch_Size, 1369, 1024] (1369 = 37x37 grid)
            outputs = self.dinov2.forward_features(input_batch)
            patch_tokens = outputs["x_norm_patchtokens"]
            
            # Chuẩn hóa vector local
            patch_tokens = F.normalize(patch_tokens, dim=2) 
        
        # 4. So sánh Local Matching
        # ref_vector: [1, 1024]
        ref_tensor = torch.from_numpy(ref_vector).to(self.device).float()
        if len(ref_tensor.shape) == 1: ref_tensor = ref_tensor.unsqueeze(0)
        ref_tensor = ref_tensor.unsqueeze(0) # [1, 1, 1024] để broadcast

        # Tính Dot Product: [Batch, 1369, 1024] * [1, 1, 1024] -> [Batch, 1369]
        # Ta nhân ref_vector với từng token trong 1369 tokens của mỗi patch
        similarity_maps = torch.sum(patch_tokens * ref_tensor, dim=2)
        
        # Lấy giá trị LỚN NHẤT trong grid 37x37 làm điểm số đại diện cho patch đó
        # Ý nghĩa: "Phần giống nhất trong cái patch này giống cái Balo bao nhiêu?"
        max_scores, _ = torch.max(similarity_maps, dim=1)
        scores = max_scores.cpu().numpy()
        
        # 5. Trả về kết quả
        detections = []
        for i, score in enumerate(scores):
            x1, y1, x2, y2 = coords[i]
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "patch_idx": i
            })
            
        detections.sort(key=lambda x: x['score'], reverse=True)
        return detections, patches_pil