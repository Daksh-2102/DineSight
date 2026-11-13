"""
DineSight AI - Object Detection Module (Module 2: Detector)
===========================================================
Responsible for loading the YOLOv8 object detection model, performing
inference on incoming video frames, filtering for specific target classes
(initially the 'person' class), and returning structured bounding box detections.

Key Capabilities:
- Automatic hardware detection (CUDA GPU acceleration if available, otherwise CPU).
- Target class filtering (default COCO class 0: 'person').
- Configurable confidence threshold and inference image size.
- Pure inference latency measurement (in milliseconds and inference FPS).
- Clean OpenCV annotation utilities for bounding boxes and confidence tags.
- Robust error handling for missing models, invalid frames, and empty detections.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8 Object Detection Engine for DineSight AI.
    """

    def __init__(
        self,
        model_path: str = "data/models/yolov8n.pt",
        conf_threshold: float = 0.35,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        """
        Initialize the YOLOv8 detector.

        Args:
            model_path: Path to the YOLO weights file (.pt).
            conf_threshold: Minimum confidence score [0.0 - 1.0] to accept a detection.
            target_classes: List of class IDs to filter for. Default [0] ('person').
            device: 'cuda', 'cpu', or None ('auto' detects available hardware).
            imgsz: Input image dimension for model inference (e.g., 640).
        """
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.target_classes = target_classes if target_classes is not None else [0]
        self.imgsz = imgsz

        # 1. Validate weights file
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[Detector Error] YOLO model weights not found at '{self.model_path}'.\n"
                f"Please ensure yolov8n.pt exists in data/models/."
            )

        # 2. Hardware Detection & Device Selection
        self.device, self.device_name = self._resolve_device(device)

        # 3. Load YOLOv8 Model
        try:
            self.model = YOLO(self.model_path)
            # Warm up or verify model classes
            self.class_names: Dict[int, str] = self.model.names
        except Exception as e:
            raise RuntimeError(f"[Detector Error] Failed to load YOLO model: {e}")

        # Performance metrics
        self.last_inference_time_ms: float = 0.0
        self.last_inference_fps: float = 0.0

    @staticmethod
    def _resolve_device(requested_device: Optional[str]) -> Tuple[str, str]:
        """
        Determine the appropriate compute device (CUDA vs CPU).
        Returns tuple: (device_str, human_readable_name).
        """
        cuda_available = torch.cuda.is_available()

        if requested_device is not None and requested_device.lower() in ("cuda", "gpu", "0"):
            if not cuda_available:
                print(
                    "[Detector Warning] CUDA was explicitly requested, but torch.cuda.is_available() is False. "
                    "Falling back to CPU."
                )
                return "cpu", "CPU"
            device_name = torch.cuda.get_device_name(0)
            return "cuda:0", f"GPU ({device_name})"

        if requested_device is not None and requested_device.lower() == "cpu":
            return "cpu", "CPU"

        # Automatic detection
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            return "cuda:0", f"GPU ({device_name})"
        else:
            return "cpu", "CPU"

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform object detection on a single frame.

        Args:
            frame: BGR image frame from OpenCV (NumPy array).

        Returns:
            List of detection dicts:
            [
                {
                    "bbox": [x1, y1, x2, y2],  # coordinates as ints
                    "confidence": 0.85,         # float score
                    "class_id": 0,              # integer ID
                    "class_name": "person",     # string label
                    "center": (cx, cy)          # bounding box center point
                },
                ...
            ]
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        t_start = time.perf_counter()

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=self.target_classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        t_end = time.perf_counter()
        delta_s = t_end - t_start
        self.last_inference_time_ms = delta_s * 1000.0
        self.last_inference_fps = 1.0 / delta_s if delta_s > 0 else 0.0

        detections: List[Dict] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            # Bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names.get(cls_id, str(cls_id))

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": cls_name,
                "center": (cx, cy),
            })

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 215, 255),  # DineSight gold
        box_thickness: int = 2,
    ) -> np.ndarray:
        """
        Annotate bounding boxes and confidence labels on a copy of the frame.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class_name"]

            # Draw bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

            # Label badge
            label = f"{cls_name.capitalize()}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            # Badge background
            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (x1 + text_w + 8, badge_y2),
                color,
                -1,
            )
            # Text inside badge (dark contrast text)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, badge_y2 - 4),
                font,
                font_scale,
                (20, 20, 20),
                thickness,
                cv2.LINE_AA,
            )

        return annotated

