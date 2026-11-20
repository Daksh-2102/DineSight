import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from backend.app.core.detector import YOLODetector
from backend.app.core.roi_manager import ROIManager

detector = YOLODetector(conf_threshold=0.20)
roi_mgr = ROIManager('config/rois.json')
cap = cv2.VideoCapture('data/sample_videos/restaurant_cctv_sample.mp4')

sample_indices = [0, 100, 250, 500, 750, 1000, 1250, 1450]
for idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    dets = detector.detect(frame)
    table_counts = {t: 0 for t in roi_mgr.tables}
    unassigned = 0
    for d in dets:
        tbl = roi_mgr.find_table_for_bbox(d['bbox'], w, h)
        if tbl:
            table_counts[tbl] += 1
        else:
            unassigned += 1
    t1 = table_counts['table_1']
    t2 = table_counts['table_2']
    t3 = table_counts['table_3']
    t4 = table_counts['table_4']
    t5 = table_counts['table_5']
    print(f"Frame {idx:4d}: {len(dets)} detections | T1:{t1} T2:{t2} T3:{t3} T4:{t4} T5:{t5} | unassigned:{unassigned}")

cap.release()
