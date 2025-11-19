# src/pipelines/temporal_filter.py

# ... (imports) ...
from src.utils.video import frame_generator # Thay đổi import

class TemporalFilter:
    def __init__(self, dino_encoder: DINOv2Encoder, config):
        self.encoder = dino_encoder
        self.config = config

    def _compute_query_vector(self, ref_images_pil):
        # ... (giữ nguyên) ...
        pass

    def find_regions_of_interest(self, video_path: str, ref_images_pil: list):
        print("Stage 1.1: Temporal Filtering (Streaming Mode)...")
        
        query_vector = self._compute_query_vector(ref_images_pil)

        # Lấy tổng số frame để hiển thị progress bar
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        similarity_scores = np.zeros(num_frames)
        
        # --- THAY ĐỔI CỐT LÕI ---
        # Sử dụng generator thay vì đọc hết video
        video_gen = frame_generator(video_path)
        
        batch_frames = []
        frame_idx = 0
        
        with tqdm(total=num_frames, desc="Generating temporal signal") as pbar:
            for frame_pil in video_gen:
                batch_frames.append(frame_pil)
                
                # Khi đủ batch hoặc hết video, xử lý batch
                if len(batch_frames) == self.config.TEMPORAL_BATCH_SIZE or frame_idx == num_frames - 1:
                    if not batch_frames:
                        break
                    
                    frame_embeddings = self.encoder.get_cls_embedding(batch_frames)
                    frame_embeddings = F.normalize(frame_embeddings, p=2, dim=1)
                    
                    scores = F.cosine_similarity(query_vector, frame_embeddings)
                    
                    # Cập nhật điểm vào mảng lớn
                    start_idx = frame_idx - len(batch_frames) + 1
                    end_idx = frame_idx + 1
                    similarity_scores[start_idx:end_idx] = scores.numpy()
                    
                    pbar.update(len(batch_frames))
                    batch_frames = [] # Reset batch
                
                frame_idx += 1

        # ... (Phần xử lý lọc và tìm đỉnh giữ nguyên) ...
        smoothed_scores = gaussian_filter1d(similarity_scores, sigma=self.config.SIGNAL_SMOOTHING_SIGMA)
        peaks, _ = find_peaks(smoothed_scores, prominence=self.config.PEAK_PROMINENCE)

        if len(peaks) == 0:
            print("No significant peaks found.")
            return [], None # Thay đổi: không trả về list frames nữa

        trois = []
        for peak in peaks:
            start_frame = max(0, peak - self.config.ROI_WINDOW_FRAMES)
            end_frame = min(num_frames - 1, peak + self.config.ROI_WINDOW_FRAMES)
            trois.append((start_frame, end_frame))
        
        print(f"Found {len(trois)} Temporal Region(s) of Interest.")
        # Thay đổi: Không trả về list frames khổng lồ nữa
        return trois, query_vector