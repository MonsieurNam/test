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

        # Tải model DINOv2 từ torch.hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()

        # DINOv2 yêu cầu một phép biến đổi cụ thể
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def get_cls_embedding(self, images):
        """
        Trích xuất embedding [CLS] từ một batch ảnh.
        Input: list các ảnh PIL.
        Output: tensor (N, C) trên CPU.
        """
        if not isinstance(images, list):
            images = [images]
            
        # Áp dụng transform và chuyển thành batch
        t_images = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # Lấy đặc trưng và chỉ giữ lại [CLS] token (token đầu tiên)
        features = self.model(t_images)
        cls_embedding = features[:, 0, :]
        return cls_embedding.cpu()

    @torch.no_grad()
    def get_dense_feature_map(self, images):
        """
        Trích xuất bản đồ đặc trưng 2D từ một batch ảnh.
        Input: list các ảnh PIL.
        Output: tensor (N, H', W', C) trên CPU.
        """
        if not isinstance(images, list):
            images = [images]
            
        t_images = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # Sử dụng get_intermediate_layers để lấy patch tokens
        # c_l_s = class token, p_t_s = patch tokens
        # vitl14 có 1024 chiều đặc trưng
        features_dict = self.model.get_intermediate_layers(t_images, n=1, return_class_token=True)
        c_l_s, p_t_s = features_dict[0]

        # Reshape patch tokens thành bản đồ 2D
        # Kích thước patch của DINOv2 là 14x14, ảnh input là 518x518 -> 37x37 patches
        batch_size = p_t_s.shape[0]
        feature_dim = p_t_s.shape[2]
        feature_map = p_t_s.reshape(batch_size, 37, 37, feature_dim)
        
        return feature_map.cpu()