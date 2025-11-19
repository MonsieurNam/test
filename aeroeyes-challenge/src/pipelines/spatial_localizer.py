# src/pipelines/spatial_localizer.py

# ... (imports) ...
import cv2

class SpatialLocalizer:
    def __init__(self, dino_encoder: DINOv2Encoder, config):
        self.encoder = dino_encoder
        self.config = config

    def find_candidates_in_regions(self, video_path: str, query_vector: torch.Tensor, trois: list):
        print("\nStage 1.2: Spatial Localization within TRoIs (Streaming Mode)...")
        candidate_boxes_by_frame = {}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video for spatial localization: {video_path}")
            return {}

        try:
            for start_frame, end_frame in trois:
                print(f"Scanning TRoI from frame {start_frame} to {end_frame}...")
                
                # Đặt con trỏ video đến frame bắt đầu của TRoI
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                frames_in_roi = []
                for _ in range(start_frame, end_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_in_roi.append(Image.fromarray(frame_rgb))

                # Xử lý các frame trong vùng này theo batch
                for i in tqdm(range(0, len(frames_in_roi), self.config.SPATIAL_BATCH_SIZE)):
                    batch_frames = frames_in_roi[i:i+self.config.SPATIAL_BATCH_SIZE]
                    if not batch_frames:
                        break
                    
                    current_frame_indices = range(start_frame + i, start_frame + i + len(batch_frames))
                    original_dims = [(f.width, f.height) for f in batch_frames]

                    # ... (Toàn bộ phần xử lý heatmap và tìm bbox giữ nguyên y hệt code cũ) ...
                    feature_maps = self.encoder.get_dense_feature_map(batch_frames)
                    feature_maps = F.normalize(feature_maps, p=2, dim=-1)
                    heatmaps = F.cosine_similarity(query_vector.view(1, 1, 1, -1), feature_maps)

                    for j, heatmap in enumerate(heatmaps):
                        frame_idx = current_frame_indices[j]
                        w, h = original_dims[j]
                        # ... (code resize, threshold, findContours, v.v...) ...
                        # ...
                        # ...
                        if boxes:
                            candidate_boxes_by_frame[frame_idx] = boxes
        finally:
            cap.release()
            
        print(f"Found candidate boxes in {len(candidate_boxes_by_frame)} frames.")
        return candidate_boxes_by_frame