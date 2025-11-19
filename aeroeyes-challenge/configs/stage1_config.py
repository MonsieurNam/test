# configs/stage1_config.py

# --- DINOv2 Configuration ---
DINO_MODEL_NAME = "dinov2_vitl14" # l=large, g=giant. Start with l for speed.

# --- Temporal Filter Configuration ---
TEMPORAL_BATCH_SIZE = 32          # Số frame xử lý song song để lấy CLS token
SIGNAL_SMOOTHING_SIGMA = 5        # Độ mịn của bộ lọc Gaussian
PEAK_PROMINENCE = 0.05            # Độ "nổi bật" của đỉnh tín hiệu để được coi là đáng chú ý
ROI_WINDOW_FRAMES = 75            # Mở rộng 75 frame về 2 phía của đỉnh để tạo TRoI (3 giây)

# --- Spatial Localizer Configuration ---
SPATIAL_BATCH_SIZE = 8           # Batch size để lấy feature map (nhỏ hơn vì tốn VRAM)
HEATMAP_THRESHOLD = 0.65         # Ngưỡng để coi một vùng là ứng cử viên

# --- Paths ---
PRETRAINED_MODELS_DIR = "./pretrained_models/"
DATASET_DIR = "./dataset/samples/"
OUTPUT_DIR = "./outputs/stage1/"