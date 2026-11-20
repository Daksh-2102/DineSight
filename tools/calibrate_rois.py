"""
DineSight AI - Interactive ROI Calibration Utility
==================================================
An interactive OpenCV tool to define and calibrate polygonal Regions of Interest
(ROIs) directly on CCTV video frames.

Features:
- Point-and-click polygon drawing on any frame of the CCTV video.
- Real-time visual feedback of vertices, edges, and area fills.
- Normalized coordinate export ([0.0 - 1.0]) for resolution independence.
- Support for Tables, Queue, Service Counter, and Entrance zones.
- Keyboard shortcuts for rapid workflow:
    [Click] : Add vertex to active polygon
    [u]/[z] : Undo last vertex
    [c]     : Clear active polygon
    [n]     : Next ROI zone
    [p]     : Previous ROI zone
    [s]     : Save all ROIs to JSON configuration file
    [q]/ESC : Quit
- Built-in default calibrated preset for standard CCTV test video.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Palette for distinct ROI visualization
ZONE_COLORS = {
    "table_1": (0, 215, 255),      # Gold
    "table_2": (255, 140, 0),      # Deep Orange
    "table_3": (50, 205, 50),      # Lime Green
    "table_4": (30, 144, 255),     # Dodger Blue
    "table_5": (238, 130, 238),    # Violet
    "queue": (0, 255, 255),        # Cyan
    "counter": (255, 105, 180),    # Hot Pink
    "entrance": (0, 165, 255),     # Orange-Red
}

# Standard zone sequence for calibration
ZONE_DEFINITIONS = [
    {"id": "table_1", "type": "table", "name": "Table 1 (Foreground Dining)", "max_capacity": 4},
    {"id": "table_2", "type": "table", "name": "Table 2 (Center Dining)", "max_capacity": 2},
    {"id": "table_3", "type": "table", "name": "Table 3 (Left Booth Foreground)", "max_capacity": 2},
    {"id": "table_4", "type": "table", "name": "Table 4 (Left Booth Background)", "max_capacity": 2},
    {"id": "table_5", "type": "table", "name": "Table 5 (Back Center / Background)", "max_capacity": 2},
    {"id": "counter", "type": "counter", "name": "Service / Ordering Counter"},
    {"id": "queue", "type": "queue", "name": "Order / Service Counter Queue"},
    {"id": "entrance", "type": "entrance", "name": "Restaurant Entrance"},
]

# High-accuracy calibrated coordinates for data/sample_videos/restaurant_cctv_sample.mp4
# Calibrated to real CCTV perspective (848x480)
CALIBRATED_PRESET = {
    "version": "1.2",
    "description": "DineSight AI - Calibrated Perspective ROI Configuration",
    "coordinate_type": "normalized",
    "tables": [
        {
            "id": "table_1",
            "name": "Table 1 (Foreground Dining)",
            "polygon": [
                [0.35, 0.66],
                [0.64, 0.66],
                [0.64, 0.98],
                [0.35, 0.98]
            ],
            "max_capacity": 4
        },
        {
            "id": "table_2",
            "name": "Table 2 (Center Dining)",
            "polygon": [
                [0.30, 0.42],
                [0.62, 0.42],
                [0.62, 0.66],
                [0.30, 0.66]
            ],
            "max_capacity": 2
        },
        {
            "id": "table_3",
            "name": "Table 3 (Left Booth Foreground)",
            "polygon": [
                [0.05, 0.40],
                [0.34, 0.40],
                [0.34, 0.96],
                [0.05, 0.96]
            ],
            "max_capacity": 2
        },
        {
            "id": "table_4",
            "name": "Table 4 (Left Booth Background)",
            "polygon": [
                [0.12, 0.30],
                [0.34, 0.30],
                [0.34, 0.42],
                [0.12, 0.42]
            ],
            "max_capacity": 2
        },
        {
            "id": "table_5",
            "name": "Table 5 (Back Center / Background)",
            "polygon": [
                [0.48, 0.28],
                [0.68, 0.28],
                [0.68, 0.42],
                [0.48, 0.42]
            ],
            "max_capacity": 2
        }
    ],
    "counter": {
        "id": "counter_zone",
        "name": "Service / Ordering Counter",
        "polygon": [
            [0.68, 0.30],
            [0.88, 0.30],
            [0.88, 0.62],
            [0.68, 0.62]
        ]
    },
    "queue": {
        "id": "queue_zone",
        "name": "Order / Service Counter Queue",
        "polygon": [
            [0.65, 0.38],
            [0.98, 0.38],
            [0.98, 0.98],
            [0.65, 0.98]
        ]
    },
    "entrance": {
        "id": "entrance_zone",
        "name": "Restaurant Entrance",
        "polygon": [
            [0.02, 0.05],
            [0.35, 0.05],
            [0.35, 0.32],
            [0.02, 0.32]
        ]
    }
}


class ROICalibrator:
    """Interactive ROI calibration manager."""

    def __init__(self, frame: np.ndarray, output_path: str = "config/rois.json"):
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.output_path = output_path
        
        self.zones = ZONE_DEFINITIONS
        self.current_zone_idx = 0
        
        # Store polygons as lists of normalized [nx, ny] coordinates
        self.polygons: Dict[str, List[List[float]]] = {z["id"]: [] for z in self.zones}
        
        # Load existing config if available to pre-populate
        self._load_existing()

    def _load_existing(self):
        """Pre-populate with existing config or calibrated preset."""
        preset = CALIBRATED_PRESET
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as fp:
                    preset = json.load(fp)
            except Exception:
                pass
        
        # Map tables
        for t in preset.get("tables", []):
            if t["id"] in self.polygons:
                self.polygons[t["id"]] = [list(pt) for pt in t.get("polygon", [])]
        
        # Map queue, entrance, counter
        for special in ["queue", "entrance", "counter"]:
            if special in preset and special in self.polygons:
                self.polygons[special] = [list(pt) for pt in preset[special].get("polygon", [])]

    def add_point(self, x: int, y: int):
        """Add pixel point, converted to normalized coordinates."""
        nx = round(x / self.w, 3)
        ny = round(y / self.h, 3)
        zone_id = self.zones[self.current_zone_idx]["id"]
        self.polygons[zone_id].append([nx, ny])

    def undo_point(self):
        """Remove last point from active polygon."""
        zone_id = self.zones[self.current_zone_idx]["id"]
        if self.polygons[zone_id]:
            self.polygons[zone_id].pop()

    def clear_active(self):
        """Clear all points from active polygon."""
        zone_id = self.zones[self.current_zone_idx]["id"]
        self.polygons[zone_id] = []

    def next_zone(self):
        """Move to next zone."""
        self.current_zone_idx = (self.current_zone_idx + 1) % len(self.zones)

    def prev_zone(self):
        """Move to previous zone."""
        self.current_zone_idx = (self.current_zone_idx - 1) % len(self.zones)

    def save(self):
        """Export current polygons to JSON configuration."""
        data = {
            "version": "1.2",
            "description": "DineSight AI - Calibrated CCTV Perspective ROI Configuration",
            "coordinate_type": "normalized",
            "tables": [],
            "counter": None,
            "queue": None,
            "entrance": None
        }

        for z in self.zones:
            zid = z["id"]
            poly = self.polygons.get(zid, [])
            if not poly:
                continue

            if z["type"] == "table":
                data["tables"].append({
                    "id": zid,
                    "name": z["name"],
                    "polygon": poly,
                    "max_capacity": z.get("max_capacity", 2)
                })
            elif z["type"] == "queue":
                data["queue"] = {
                    "id": f"{zid}_zone",
                    "name": z["name"],
                    "polygon": poly
                }
            elif z["type"] == "counter":
                data["counter"] = {
                    "id": f"{zid}_zone",
                    "name": z["name"],
                    "polygon": poly
                }
            elif z["type"] == "entrance":
                data["entrance"] = {
                    "id": f"{zid}_zone",
                    "name": z["name"],
                    "polygon": poly
                }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as fp:
            json.dump(data, fp, indent=2)
        print(f"[Calibrator] Configuration successfully saved to: {self.output_path}")

    def render(self) -> np.ndarray:
        """Render annotated calibration display."""
        vis = self.frame.copy()
        overlay = self.frame.copy()

        active_zone = self.zones[self.current_zone_idx]
        active_id = active_zone["id"]

        # Draw all polygons
        for z in self.zones:
            zid = z["id"]
            pts = self.polygons.get(zid, [])
            if not pts:
                continue

            # Convert normalized to pixel coordinates
            pixel_pts = np.array([[int(p[0] * self.w), int(p[1] * self.h)] for p in pts], dtype=np.int32)
            color = ZONE_COLORS.get(zid, (200, 200, 200))
            is_active = (zid == active_id)

            # Draw polygon fill on overlay
            if len(pixel_pts) >= 3:
                cv2.fillPoly(overlay, [pixel_pts], color)

            # Draw perimeter lines
            thickness = 3 if is_active else 1
            if len(pixel_pts) >= 2:
                cv2.polylines(vis, [pixel_pts], isClosed=True if len(pixel_pts) >= 3 else False, color=color, thickness=thickness)

            # Draw vertices
            for pt in pixel_pts:
                cv2.circle(vis, tuple(pt), 4 if is_active else 2, (0, 0, 255) if is_active else color, -1)

            # Label centroid
            if len(pixel_pts) >= 3:
                cx = int(np.mean(pixel_pts[:, 0]))
                cy = int(np.mean(pixel_pts[:, 1]))
                label = z["name"].split(" ")[0]
                cv2.putText(vis, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend overlay (alpha = 0.25)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # Top Banner HUD
        cv2.rectangle(vis, (0, 0), (self.w, 40), (20, 20, 20), -1)
        cur_color = ZONE_COLORS.get(active_id, (0, 215, 255))
        cv2.rectangle(vis, (10, 8), (22, 28), cur_color, -1)
        
        info_text = f"ZONE: {active_zone['name']} ({len(self.polygons[active_id])} pts) | [N]ext [P]rev [U]ndo [C]lear [S]ave [Q]uit"
        cv2.putText(vis, info_text, (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        return vis


def run_interactive_calibration(video_path: str, output_path: str = "config/rois.json", frame_idx: int = 100):
    """Run interactive calibration loop."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not load frame {frame_idx} from {video_path}")

    calibrator = ROICalibrator(frame, output_path=output_path)
    window_name = "DineSight AI - ROI Calibration Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            calibrator.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            calibrator.undo_point()

    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 70)
    print("       DINESIGHT AI - INTERACTIVE ROI CALIBRATION UTILITY")
    print("=" * 70)
    print("Controls:")
    print("  [Left Click]   : Add polygon vertex")
    print("  [Right Click]  : Undo last vertex")
    print("  [n]            : Next ROI zone")
    print("  [p]            : Previous ROI zone")
    print("  [u]            : Undo point")
    print("  [c]            : Clear current zone polygon")
    print("  [s]            : Save all ROIs to config/rois.json")
    print("  [q] or [ESC]   : Exit calibrator\n")

    while True:
        display = calibrator.render()
        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            calibrator.next_zone()
        elif key == ord('p'):
            calibrator.prev_zone()
        elif key in (ord('u'), ord('z')):
            calibrator.undo_point()
        elif key == ord('c'):
            calibrator.clear_active()
        elif key == ord('s'):
            calibrator.save()

    cv2.destroyAllWindows()


def apply_calibrated_preset(output_path: str = "config/rois.json"):
    """Directly save high-accuracy calibrated preset without GUI."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(CALIBRATED_PRESET, fp, indent=2)
    print(f"[Calibrator] Calibrated perspective preset applied to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DineSight AI - ROI Calibration Utility")
    parser.add_argument("--source", type=str, default="data/sample_videos/restaurant_cctv_sample.mp4", help="Video source")
    parser.add_argument("--output", type=str, default="config/rois.json", help="Output JSON path")
    parser.add_argument("--frame", type=int, default=100, help="Frame index to calibrate on")
    parser.add_argument("--preset", action="store_true", help="Apply calibrated preset directly without opening GUI")

    args = parser.parse_args()

    if args.preset:
        apply_calibrated_preset(args.output)
    else:
        run_interactive_calibration(args.source, output_path=args.output, frame_idx=args.frame)


if __name__ == "__main__":
    main()
