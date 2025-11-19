```
# 1. Cài các thư viện cơ bản
pip install -r requirements.txt

# 2. Cài đặt Grounding DINO từ Source (Quan trọng: Máy cần có CUDA compiler)
# Nếu bạn dùng Windows, bước này có thể phức tạp, ưu tiên dùng Linux/WSL/Colab
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# 3. Cài đặt Florence-2 (nó nằm trong transformers nhưng cần bản mới nhất)
pip install --upgrade transformers

!wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```