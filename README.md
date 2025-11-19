# 1. Tạo môi trường ảo (nếu chưa có)
conda create -n aeroeyes python=3.10 -y
conda activate aeroeyes

# 2. Cài đặt PyTorch (Phiên bản hỗ trợ CUDA là BẮT BUỘC cho DINOv2/SAM2)
# Truy cập https://pytorch.org/get-started/locally/ để lấy lệnh đúng với CUDA của máy bạn
# Ví dụ cho CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Cài đặt SAM 2 (Segment Anything Model 2) từ Facebook Research
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# 4. Cài đặt các thư viện hỗ trợ khác
pip install transformers  # Cho DINOv2
pip install opencv-python matplotlib pandas tqdm sahi
pip install jupyterlab    # Để debug trên notebook