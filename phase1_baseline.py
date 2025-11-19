import os
import json
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from groundingdino.util.inference import load_model, load_image, predict
import supervision as sv
from src.metric import calculate_st_iou

# --- CẤU HÌNH ---
DATASET_DIR = "dataset"
SAMPLES_DIR = os.path.join(DATASET_DIR, "samples")
ANNOTATION_FILE = os.path.join(DATASET_DIR, "annotations/annotations.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config Grounding DINO (Tự động tải weights nếu chưa có)
GD_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" # Bạn cần clone repo GD về hoặc chỉ định path đúng
# Để đơn giản cho GĐ1, ta giả định user đã có file config và weight. 
# Nếu chưa, script nên download tự động (bỏ qua ở đây để code gọn).
GD_WEIGHT_PATH = "weights/groundingdino_swint_ogc.pth" 

# --- LOAD MODELS ---
print(">>> Loading Florence-2 for Captioning...")
florence_id = 'microsoft/Florence-2-base'
florence_model = AutoModelForCausalLM.from_pretrained(florence_id, trust_remote_code=True).to(DEVICE).eval()
florence_processor = AutoProcessor.from_pretrained(florence_id, trust_remote_code=True)

print(">>> Loading Grounding DINO for Detection...")
# Lưu ý: Bạn cần file config .py của GroundingDINO
# Nếu dùng thư viện python installed, đoạn này có thể khác. Đây là dùng theo repo gốc.
# gd_model = load_model(GD_CONFIG_PATH, GD_WEIGHT_PATH) 
# TẠM THỜI: Giả lập load model để code chạy logic (Vì path config phụ thuộc máy bạn)
# Bạn hãy thay thế bằng hàm load thực tế khi chạy
gd_model = None # Placeholder

def generate_prompt(image_path):
    """Dùng Florence-2 sinh prompt từ ảnh reference"""
    image = Image.open(image_path).convert("RGB")
    task = "<CAPTION>"
    inputs = florence_processor(text=task, images=image, return_tensors="pt").to(DEVICE, torch.float16)
    
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            num_beams=3
        )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = florence_processor.post_process_generation(generated_text, task=task, image_size=image.size)
    
    # Xử lý string đơn giản: lấy danh từ chính (ví dụ sơ khai)
    caption = parsed[task]
    # Giai đoạn 1: Dùng luôn caption làm prompt. 
    # VD: "A grey backpack on the ground" -> Grounding DINO hiểu tốt.
    return caption

def process_video(video_id, folder_path, gd_model):
    # 1. Lấy ảnh Reference và sinh Prompt
    ref_img_path = os.path.join(folder_path, "object_images", "img_1.jpg") # Lấy đại ảnh số 1
    text_prompt = generate_prompt(ref_img_path)
    print(f"[{video_id}] Prompt generated: {text_prompt}")
    
    # 2. Xử lý Video
    video_path = os.path.join(folder_path, "drone_video.mp4")
    cap = cv2.VideoCapture(video_path)
    
    detections_dict = {} # Format: {frame_idx: [x1, y1, x2, y2]}
    frame_idx = 0
    
    # BOX THRESHOLD: Quan trọng!
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # OPTIMIZATION: Chỉ detect mỗi 5 frame để test nhanh
        if frame_idx % 5 == 0:
            # Convert frame cv2 (BGR) -> PIL (RGB)
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Grounding DINO Inference
            # Lưu ý: Hàm predict này cần model thật. 
            # Ở GĐ1, nếu chưa cài xong GD, code sẽ dừng ở đây.
            if gd_model: 
                boxes, logits, phrases = predict(
                    model=gd_model, 
                    image=image_pil, 
                    caption=text_prompt, 
                    box_threshold=BOX_THRESHOLD, 
                    text_threshold=TEXT_THRESHOLD
                )
                
                # Xử lý kết quả: Lấy box có score cao nhất
                if len(boxes) > 0:
                    # boxes trả về dạng normalized [cx, cy, w, h] -> Cần đổi sang [x1, y1, x2, y2] absolute
                    h, w, _ = frame.shape
                    boxes_xyxy = sv.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
                    
                    # Lấy box đầu tiên (có score cao nhất)
                    best_box = boxes_xyxy[0].numpy().tolist()
                    detections_dict[frame_idx] = best_box
        
        frame_idx += 1
    cap.release()
    return detections_dict

def main():
    # Load Ground Truth để tính điểm
    with open(ANNOTATION_FILE, 'r') as f:
        gt_data = json.load(f)
    
    # Convert GT sang dict để dễ tra cứu: gt_lookup[video_id][frame] = box
    gt_lookup = {}
    for item in gt_data:
        vid = item['video_id']
        gt_lookup[vid] = {}
        for ann in item['annotations']:
             for b in ann['bboxes']:
                 gt_lookup[vid][b['frame']] = [b['x1'], b['y1'], b['x2'], b['y2']]

    final_submission = []
    total_st_iou = 0
    video_count = 0

    # Duyệt qua từng folder video trong dataset/samples
    video_folders = sorted(os.listdir(SAMPLES_DIR))
    
    # Load GD Model thật ở đây (bạn cần trỏ đúng path config/weight)
    # gd_model = load_model(...) 
    
    for vid_folder in tqdm(video_folders):
        # Giả sử tên folder trùng tên video_id hoặc cần mapping
        video_id = vid_folder # Cần check lại mapping với file annotations.json
        full_path = os.path.join(SAMPLES_DIR, vid_folder)
        
        if not os.path.isdir(full_path): continue
        
        # --- RUN PIPELINE ---
        pred_detections = process_video(video_id, full_path, gd_model)
        
        # --- TÍNH ST-IoU ---
        # Cần mapping tên video_id trong folder với video_id trong json annotation
        # Nếu folder tên "drone_video_001" nhưng json là "drone_video_01", phải xử lý string
        # Tạm thời assume tên trùng nhau
        if video_id in gt_lookup:
            score = calculate_st_iou(pred_detections, gt_lookup[video_id])
            print(f"Video {video_id} - ST-IoU: {score:.4f}")
            total_st_iou += score
            video_count += 1
        
        # Format output submission
        # Chuyển format detections_dict về format list của cuộc thi
        det_list = []
        for f_idx, box in pred_detections.items():
            det_list.append({
                "frame": f_idx,
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]
            })
        
        final_submission.append({
            "video_id": video_id,
            "detections": [{"bboxes": det_list}] if det_list else []
        })

    print(f"=== AVERAGE ST-IoU: {total_st_iou/video_count if video_count > 0 else 0} ===")
    
    # Lưu file submission
    with open("submission.json", "w") as f:
        json.dump(final_submission, f)

if __name__ == "__main__":
    main()