"""
DineSight AI - Multi-Object Tracking Module (Module 2: Tracker)
==============================================================
Responsible for tracking detected individuals across video frames using the
ByteTrack algorithm.

Key Principles:
- Temporal Association: Associates detections across consecutive frames to assign
  persistent, temporary integer tracking IDs (e.g. ID #1, ID #2).
- Ephemeral Identifiers: Tracking IDs are maintained in memory solely during video
  processing and are NEVER stored as permanent customer identities.
- Privacy-by-Design: No facial recognition or biometric profiles are constructed.
- Handles entry, movement, and departure of individuals seamlessly.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.app.core.detector import COCO_TABLEWARE_CLASSES, COCO_TABLEWARE_NAMES


# Distinct visual palette for track ID bounding boxes
TRACK_COLORS = [
    (0, 215, 255),   # DineSight Gold
    (50, 205, 50),   # Lime Green
    (255, 140, 0),   # Deep Orange
    (30, 144, 255),  # Dodger Blue
    (238, 130, 238), # Violet
    (0, 255, 255),   # Cyan
    (255, 105, 180), # Hot Pink
    (147, 112, 219), # Medium Purple
    (72, 209, 204),  # Turquoise
    (255, 215, 0),   # Yellow
]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR color for a given integer track ID."""
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker Wrapper for DineSight AI.
    
    Uses Ultralytics YOLOv8's built-in ByteTrack integration to detect
    and track target objects across consecutive video frames.
    """

    DEFAULT_TRACKER_CONFIG = "config/bytetrack_dinesight.yaml"

    def __init__(
        self,
        model_path: str = "data/models/yolov8n.pt",
        conf_threshold: float = 0.20,
        target_classes: Optional[List[int]] = None,
        tracker_type: Optional[str] = None,
        device: Optional[str] = None,
        imgsz: int = 640,
        include_tableware: bool = False,
    ):
        """
        Initialize the ByteTracker.

        Args:
            model_path: Path to YOLO weights file (.pt).
            conf_threshold: Minimum detection confidence score. Calibrated to 0.20
                            to allow ByteTrack two-stage association to match
                            partially occluded seated patrons.
            target_classes: List of class IDs to track. Default [0] ('person').
            tracker_type: Tracker configuration file path or name. If None,
                          automatically resolves config/bytetrack_dinesight.yaml
                          if present, falling back to built-in 'bytetrack.yaml'.
            device: 'cuda', 'cpu', or None (auto-detect).
            imgsz: Input resolution for inference.
            include_tableware: If True and target_classes is None, tracks both persons
                               and COCO tableware items (bottles, cups, bowls, cutlery).
        """
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        if target_classes is not None:
            self.target_classes = target_classes
        elif include_tableware:
            self.target_classes = [0] + COCO_TABLEWARE_CLASSES
        else:
            self.target_classes = [0]
        self.imgsz = imgsz

        # Resolve tracker configuration
        if tracker_type is not None:
            self.tracker_type = tracker_type
        elif os.path.exists(self.DEFAULT_TRACKER_CONFIG):
            self.tracker_type = self.DEFAULT_TRACKER_CONFIG
        else:
            self.tracker_type = "bytetrack.yaml"

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[Tracker Error] Model weights not found at '{self.model_path}'."
            )

        # Device resolution
        cuda_available = torch.cuda.is_available()
        if device is not None and device.lower() in ("cuda", "gpu", "0"):
            self.device = "cuda:0" if cuda_available else "cpu"
            self.device_name = f"GPU ({torch.cuda.get_device_name(0)})" if cuda_available else "CPU"
        elif device is not None and device.lower() == "cpu":
            self.device = "cpu"
            self.device_name = "CPU"
        else:
            if cuda_available:
                self.device = "cuda:0"
                self.device_name = f"GPU ({torch.cuda.get_device_name(0)})"
            else:
                self.device = "cpu"
                self.device_name = "CPU"

        # Load YOLO model
        try:
            self.model = YOLO(self.model_path)
            self.class_names: Dict[int, str] = self.model.names
        except Exception as e:
            raise RuntimeError(f"[Tracker Error] Failed to initialize model for tracking: {e}")

        # Metrics
        self.last_track_time_ms: float = 0.0
        self.last_track_fps: float = 0.0

        # Memory of unique track IDs seen in this session (transient only)
        self.active_track_ids: set = set()
        self.total_unique_tracks_seen: int = 0
        # In-memory track telemetry (transient, non-biometric, privacy-preserving)
        self.track_metadata: Dict[int, Dict] = {}

    def set_confidence_threshold(self, conf: float) -> None:
        """Dynamically update tracker detection confidence threshold [0.0 - 1.0]."""
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {conf}")
        self.conf_threshold = float(conf)

    def set_tracker_type(self, tracker_type: str) -> None:
        """Update tracker configuration file."""
        self.tracker_type = tracker_type

    def get_track_duration(self, track_id: int) -> float:
        """Return duration in seconds this track has been continuously observed."""
        meta = self.track_metadata.get(track_id)
        if not meta:
            return 0.0
        return meta["last_seen"] - meta["first_seen"]

    @staticmethod
    def separate_tracks(tracks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split a combined track list into human person tracks and tableware items.

        Returns:
            Tuple of (person_tracks, tableware_tracks)
        """
        person_tracks = []
        tableware_tracks = []
        for trk in tracks:
            cls_id = trk.get("class_id", 0)
            if cls_id == 0:
                person_tracks.append(trk)
            elif cls_id in COCO_TABLEWARE_CLASSES:
                tableware_tracks.append(trk)
            else:
                person_tracks.append(trk)
        return person_tracks, tableware_tracks

    def update(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a video frame, update ByteTrack state, and return tracked objects.

        Args:
            frame: BGR image frame (NumPy array).

        Returns:
            List of tracked object dictionaries:
            [
                {
                    "track_id": 1,              # Persistent temporary tracking ID
                    "bbox": [x1, y1, x2, y2],   # Bounding box coordinates
                    "confidence": 0.88,          # Detection confidence
                    "class_id": 0,               # Target class ID
                    "class_name": "person",      # Class label
                    "center": (cx, cy)           # Centroid point
                },
                ...
            ]
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        t_start = time.perf_counter()

        # Run ByteTrack multi-object tracking via Ultralytics
        # persist=True tells the tracker that this frame is part of an ongoing video stream
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.conf_threshold,
            classes=self.target_classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        t_end = time.perf_counter()
        delta_s = t_end - t_start
        self.last_track_time_ms = delta_s * 1000.0
        self.last_track_fps = 1.0 / delta_s if delta_s > 0 else 0.0

        tracks: List[Dict] = []
        if not results or len(results) == 0:
            return tracks

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return tracks

        current_frame_ids = set()

        for box in boxes:
            # Extract coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names.get(cls_id, str(cls_id))

            # Extract assigned ByteTrack ID (if tracker assigned one)
            if box.id is not None:
                track_id = int(box.id[0].cpu().numpy())
            else:
                # If a new low-confidence detection hasn't yet formed a tracklet,
                # we don't assign an arbitrary ID to prevent false count inflation
                continue

            current_frame_ids.add(track_id)
            now_t = time.time()
            if track_id not in self.active_track_ids:
                self.active_track_ids.add(track_id)
                self.total_unique_tracks_seen += 1
                self.track_metadata[track_id] = {
                    "first_seen": now_t,
                    "last_seen": now_t,
                    "frame_count": 1,
                }
            else:
                self.track_metadata[track_id]["last_seen"] = now_t
                self.track_metadata[track_id]["frame_count"] += 1

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            tracks.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": cls_name,
                "center": (cx, cy),
            })

        return tracks

    def reset(self) -> None:
        """
        Reset tracker state when video is rewound or reloaded.
        """
        self.active_track_ids.clear()
        self.total_unique_tracks_seen = 0
        self.track_metadata.clear()
        # Re-initializing the YOLO model predictor clears internal tracker buffers
        try:
            self.model = YOLO(self.model_path)
        except Exception:
            pass

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        show_center: bool = True,
        box_thickness: int = 2,
    ) -> np.ndarray:
        """
        Annotate bounding boxes, persistent track IDs, and confidence tags on a frame.
        """
        annotated = frame.copy()

        for trk in tracks:
            track_id = trk["track_id"]
            x1, y1, x2, y2 = trk["bbox"]
            conf = trk["confidence"]
            cls_name = trk["class_name"]
            cx, cy = trk["center"]

            cls_id = trk.get("class_id", 0)
            is_tableware = cls_id in COCO_TABLEWARE_CLASSES
            color = (255, 140, 0) if is_tableware else get_color_for_id(track_id)
            active_thickness = 1 if is_tableware else box_thickness

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, active_thickness)

            # Draw center centroid dot
            if show_center and not is_tableware:
                cv2.circle(annotated, (cx, cy), 4, color, -1)
                cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 1)

            # Construct label: "Person #1 (85%)" or "Bottle #4 (37%)"
            label = f"{cls_name.capitalize()} #{track_id} ({int(conf * 100)}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38 if is_tableware else 0.45
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # If person is near the top edge or HUD, flip badge inside the bounding box
            if y1 < 78:
                badge_y1 = y1 + 2
                badge_y2 = badge_y1 + text_h + 6
            else:
                badge_y1 = max(0, y1 - text_h - 6)
                badge_y2 = y1

            # High-contrast dark badge with colored border
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (x1 + text_w + 8, badge_y2),
                (18, 22, 28),
                -1,
            )
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (x1 + text_w + 8, badge_y2),
                color,
                1,
            )
            # Text inside badge
            cv2.putText(
                annotated,
                label,
                (x1 + 4, badge_y2 - 4),
                font,
                font_scale,
                (240, 240, 240) if not is_tableware else (255, 180, 50),
                thickness,
                cv2.LINE_AA,
            )

        return annotated

