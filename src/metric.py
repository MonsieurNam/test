# src/metric.py
import numpy as np

def calculate_iou(box1, box2):
    """
    Tính Spatial IoU giữa 2 box [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union

def calculate_st_iou(pred_bboxes, gt_bboxes):
    """
    Tính Spatio-Temporal IoU cho 1 video.
    Input:
        pred_bboxes: dict {frame_idx: [x1, y1, x2, y2]}
        gt_bboxes: dict {frame_idx: [x1, y1, x2, y2]}
    """
    # Lấy tập hợp các frame xuất hiện trong cả GT và Pred
    gt_frames = set(gt_bboxes.keys())
    pred_frames = set(pred_bboxes.keys())
    
    intersection_frames = gt_frames.intersection(pred_frames)
    union_frames = gt_frames.union(pred_frames)
    
    if len(union_frames) == 0:
        return 0.0

    # Tử số: Tổng IoU của các frame giao nhau
    sum_iou = 0.0
    for f in intersection_frames:
        sum_iou += calculate_iou(pred_bboxes[f], gt_bboxes[f])
    
    # Mẫu số: Tổng số lượng frame trong tập hợp (Union)
    # Theo công thức: Sum(1 for f in union) chính là len(union_frames)
    st_iou = sum_iou / len(union_frames)
    
    return st_iou