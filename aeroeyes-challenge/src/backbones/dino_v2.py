# src/backbones/dino_v2.py

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

class DINOv2Encoder:
    """
    Một lớp wrapper để dễ dàng sử dụng DINOv2 cho việc trích xuất đặc trưng.
    """
    def __init__(self, model_name='dinov2_vitl14', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"DINOv2Encoder is using device: {self.device}")

        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        # Lấy thông số của model để tính toán shape
        self.patch_size = self.model.patch_size
        self.grid_size = 518 // self.patch_size

    @torch.no_grad()
    def get_cls_embedding(self, images):
        """
        Trích xuất embedding [CLS] từ một batch ảnh.
        """
        if not isinstance(images, list):
            images = [images]
            
        t_images = torch.stack([self.transform(img) for img in images]).to(self.device)
        cls_embedding = self.model(t_images)
        return cls_embedding.cpu()

    @torch.no_grad()
    def get_dense_feature_map(self, images):
        """
        Trích xuất bản đồ đặc trưng 2D từ một batch ảnh.
        Sử dụng một phương pháp ổn định hơn để lấy patch tokens.
        """
        if not isinstance(images, list):
            images = [images]
            
        t_images = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # --- SỬA LỖI ---
        # Sử dụng model.forward_features để lấy cả CLS và patch tokens
        # Output là một dictionary {'x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm'}
        features_dict = self.model.forward_features(t_images)
        
        # Lấy patch tokens từ dictionary
        patch_tokens = features_dict['x_norm_patchtokens']
        
        # Reshape patch tokens thành bản đồ 2D
        batch_size = patch_tokens.shape[0]
        feature_dim = patch_tokens.shape[2]
        
        # Đảm bảo grid_size được tính toán chính xác
        # grid_size = t_images.shape[2] // self.patch_size
        feature_map = patch_tokens.reshape(batch_size, self.grid_size, self.grid_size, feature_dim)
        
        return feature_map.cpu()