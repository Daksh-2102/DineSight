"""
DineSight AI - ROI Configuration & Spatial Association Engine (Module 3A)
========================================================================
Manages 2D spatial Regions of Interest (ROIs) for restaurant tables,
entrance vestibules, and waiting queue areas.

Key Features:
- Polygon-based spatial definitions (arbitrary shapes, not just rectangles).
- Resolution-independent normalized coordinates ([0.0, 1.0]).
- Dynamic scaling to any video frame dimension (e.g. 720p, 1080p, 4K).
- Point-in-polygon association via OpenCV `cv2.pointPolygonTest`.
- Configurable person representative points (bottom-center / centroid).
- Visual polygon overlays with semi-transparent fills and status badges.
- Strict validation of coordinate bounds and polygon convexity/points.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# Color Definitions (BGR Format for OpenCV)
STATE_COLORS = {
    "VACANT": (46, 184, 46),          # Emerald Green
    "OCCUPIED": (40, 40, 220),         # Vibrant Red
    "WAITING_FOR_FOOD": (20, 140, 255),# Warm Amber / Orange
    "DINING": (220, 120, 20),          # Royal Blue
    "QUEUE": (230, 180, 40),           # Cyan / Gold
    "ENTRANCE": (160, 160, 160),       # Cool Slate Gray
    "DEFAULT": (180, 180, 180),        # Neutral Gray
}


class RegionOfInterest:
    """
    Represents a single named spatial polygon region.
    """

    def __init__(
        self,
        region_id: str,
        name: str,
        polygon_normalized: List[List[float]],
        region_type: str = "table",
        max_capacity: int = 2,
    ):
        """
        Args:
            region_id: Unique string identifier (e.g. 'table_1', 'queue_zone').
            name: Human-readable display label (e.g. 'Table 1 (2-Seater)').
            polygon_normalized: List of [x, y] coordinates in [0.0, 1.0].
            region_type: 'table', 'queue', or 'entrance'.
            max_capacity: Seating capacity (for tables).
        """
        self.region_id = str(region_id)
        self.name = str(name)
        self.region_type = region_type.lower()
        self.max_capacity = int(max_capacity)

        # Validate coordinates
        self.polygon_normalized = self._validate_and_sanitize(polygon_normalized)

        # Cached pixel polygon array for current frame dimensions
        self._cached_pixel_poly: Optional[np.ndarray] = None
        self._cached_dims: Tuple[int, int] = (0, 0)

    @staticmethod
    def _validate_and_sanitize(points: List[List[float]]) -> List[Tuple[float, float]]:
        """Validate normalized polygon coordinates."""
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            raise ValueError(f"Polygon must contain at least 3 vertices. Got {len(points) if isinstance(points, (list, tuple)) else 'non-list'}.")

        sanitized = []
        for idx, pt in enumerate(points):
            if len(pt) != 2:
                raise ValueError(f"Vertex at index {idx} must be a 2D coordinate [x, y].")
            x, y = float(pt[0]), float(pt[1])
            # Clamp or validate range [0.0, 1.0]
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"Normalized vertex [{x}, {y}] at index {idx} exceeds range [0.0, 1.0].")
            sanitized.append((x, y))
        return sanitized

    def get_pixel_polygon(self, frame_width: int, frame_height: int) -> np.ndarray:
        """
        Convert normalized vertices into absolute pixel coordinate array.
        Returns a NumPy array of shape (N, 1, 2) of type np.int32 for OpenCV.
        """
        if self._cached_pixel_poly is not None and self._cached_dims == (frame_width, frame_height):
            return self._cached_pixel_poly

        pixel_pts = []
        for nx, ny in self.polygon_normalized:
            px = int(round(nx * frame_width))
            py = int(round(ny * frame_height))
            # Clamp to frame edges
            px = max(0, min(frame_width - 1, px))
            py = max(0, min(frame_height - 1, py))
            pixel_pts.append([px, py])

        poly_arr = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))
        self._cached_pixel_poly = poly_arr
        self._cached_dims = (frame_width, frame_height)
        return poly_arr

    def contains_point(self, point: Tuple[int, int], frame_width: int, frame_height: int) -> bool:
        """
        Check if a pixel point (x, y) falls inside or on the boundary of this polygon.
        Uses OpenCV's cv2.pointPolygonTest (Ray Casting algorithm under the hood).
        """
        poly = self.get_pixel_polygon(frame_width, frame_height)
        # measureDist=False returns +1 for inside, 0 for on-edge, -1 for outside
        result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), measureDist=False)
        return result >= 0

    def get_centroid(self, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Compute the geometric centroid of the polygon in pixel coordinates."""
        poly = self.get_pixel_polygon(frame_width, frame_height)
        moments = cv2.moments(poly)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return cx, cy
        pts = poly.reshape(-1, 2)
        return int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))


class ROIManager:
    """
    Coordinates all spatial regions for DineSight AI:
    - Tables (e.g. Table 1, Table 2, Table 3)
    - Queue monitoring zone
    - Entrance area
    """

    def __init__(
        self,
        config_path: str = "config/rois.json",
        point_mode: str = "bottom_center",
    ):
        """
        Args:
            config_path: Path to JSON configuration file.
            point_mode: How to extract representative point from a person bounding box.
                        'bottom_center': ((x1+x2)//2, y2) -> Best for floor/table projection.
                        'center': ((x1+x2)//2, (y1+y2)//2) -> Center of mass.
        """
        self.config_path = config_path
        self.point_mode = point_mode.lower()
        if self.point_mode not in ("bottom_center", "center"):
            self.point_mode = "bottom_center"

        self.tables: Dict[str, RegionOfInterest] = {}
        self.queue_region: Optional[RegionOfInterest] = None
        self.entrance_region: Optional[RegionOfInterest] = None

        self.load_config(self.config_path)

    def load_config(self, filepath: str) -> None:
        """Load and parse the ROI JSON configuration file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"[ROIManager Error] ROI configuration file not found at: '{filepath}'.\n"
                f"Please ensure config/rois.json exists."
            )

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.tables.clear()
        self.queue_region = None
        self.entrance_region = None

        # 1. Parse Tables
        tables_data = data.get("tables", [])
        for t_data in tables_data:
            t_id = t_data["id"]
            name = t_data.get("name", t_id)
            poly = t_data["polygon"]
            cap = t_data.get("max_capacity", 2)
            self.tables[t_id] = RegionOfInterest(
                region_id=t_id,
                name=name,
                polygon_normalized=poly,
                region_type="table",
                max_capacity=cap,
            )

        # 2. Parse Queue
        if "queue" in data and data["queue"]:
            q_data = data["queue"]
            self.queue_region = RegionOfInterest(
                region_id=q_data.get("id", "queue_zone"),
                name=q_data.get("name", "Queue Zone"),
                polygon_normalized=q_data["polygon"],
                region_type="queue",
            )

        # 3. Parse Entrance
        if "entrance" in data and data["entrance"]:
            e_data = data["entrance"]
            self.entrance_region = RegionOfInterest(
                region_id=e_data.get("id", "entrance_zone"),
                name=e_data.get("name", "Entrance Zone"),
                polygon_normalized=e_data["polygon"],
                region_type="entrance",
            )

    def get_representative_point(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Extract a single 2D representative point (x, y) from a bounding box [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        if self.point_mode == "bottom_center":
            # For 2.5D camera perspective, the bottom-center is where the person
            # contacts the ground plane or chair.
            return cx, int(y2)
        else:  # "center"
            cy = int((y1 + y2) / 2)
            return cx, cy

    def find_table_for_point(
        self,
        point: Tuple[int, int],
        frame_width: int,
        frame_height: int,
    ) -> Optional[str]:
        """
        Find which table contains the given pixel point.
        If overlapping, returns the first match or closest centroid.
        """
        matching_tables = []
        for t_id, table in self.tables.items():
            if table.contains_point(point, frame_width, frame_height):
                matching_tables.append(t_id)

        if not matching_tables:
            return None
        if len(matching_tables) == 1:
            return matching_tables[0]

        # Tie-breaker for overlapping ROIs: choose table whose centroid is closest to the point
        best_table = None
        min_dist = float("inf")
        px, py = point
        for t_id in matching_tables:
            cx, cy = self.tables[t_id].get_centroid(frame_width, frame_height)
            dist = (cx - px) ** 2 + (cy - py) ** 2
            if dist < min_dist:
                min_dist = dist
                best_table = t_id
        return best_table

    def is_in_queue(self, point: Tuple[int, int], frame_width: int, frame_height: int) -> bool:
        """Check if a point is inside the queue zone."""
        if self.queue_region is None:
            return False
        return self.queue_region.contains_point(point, frame_width, frame_height)

    def is_in_entrance(self, point: Tuple[int, int], frame_width: int, frame_height: int) -> bool:
        """Check if a point is inside the entrance zone."""
        if self.entrance_region is None:
            return False
        return self.entrance_region.contains_point(point, frame_width, frame_height)

    def associate_tracks(
        self,
        tracks: List[Dict],
        frame_width: int,
        frame_height: int,
    ) -> Dict:
        """
        Perform spatial mapping: takes all current tracked people and associates
        them with tables, queue, or unassigned floor area.

        Returns:
            {
                "tables": {
                    "table_1": [track_obj1, ...],
                    "table_2": [],
                    ...
                },
                "queue": [track_obj3, ...],
                "entrance": [],
                "unassigned": [track_obj4, ...]
            }
        """
        mapping: Dict[str, Union[Dict[str, List[Dict]], List[Dict]]] = {
            "tables": {t_id: [] for t_id in self.tables},
            "queue": [],
            "entrance": [],
            "unassigned": [],
        }

        for trk in tracks:
            pt = self.get_representative_point(trk["bbox"])
            trk_with_pt = {**trk, "representative_point": pt}

            # 1. Check Table ROIs
            matched_table = self.find_table_for_point(pt, frame_width, frame_height)
            if matched_table is not None:
                mapping["tables"][matched_table].append(trk_with_pt)
                continue

            # 2. Check Queue ROI
            if self.is_in_queue(pt, frame_width, frame_height):
                mapping["queue"].append(trk_with_pt)
                continue

            # 3. Check Entrance ROI
            if self.is_in_entrance(pt, frame_width, frame_height):
                mapping["entrance"].append(trk_with_pt)
                continue

            # 4. Otherwise Unassigned / Walking in Aisles
            mapping["unassigned"].append(trk_with_pt)

        return mapping

    def draw_rois(
        self,
        frame: np.ndarray,
        table_states: Optional[Dict[str, str]] = None,
        table_occupant_counts: Optional[Dict[str, int]] = None,
        queue_count: int = 0,
        alpha: float = 0.22,
    ) -> np.ndarray:
        """
        Render semi-transparent colored polygons, perimeter outlines, and labels
        for all active restaurant ROIs.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Draw Dining Tables
        for t_id, table in self.tables.items():
            poly = table.get_pixel_polygon(w, h)
            state = (table_states or {}).get(t_id, "VACANT").upper()
            color = STATE_COLORS.get(state, STATE_COLORS["VACANT"])

            # Semi-transparent polygon fill
            cv2.fillPoly(overlay, [poly], color)

            # Solid perimeter border
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Label Header Badge
            cx, cy = table.get_centroid(w, h)
            occupants = (table_occupant_counts or {}).get(t_id, 0)
            label = f"{table.name}: {state}"
            if occupants > 0:
                label += f" ({occupants}p)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            badge_x1 = max(5, cx - tw // 2 - 6)
            badge_y1 = max(5, cy - th // 2 - 4)
            badge_x2 = badge_x1 + tw + 12
            badge_y2 = badge_y1 + th + 8

            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 22, 28), -1)
            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)
            cv2.putText(
                frame,
                label,
                (badge_x1 + 6, badge_y2 - 5),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        # 2. Draw Queue Zone
        if self.queue_region is not None:
            q_poly = self.queue_region.get_pixel_polygon(w, h)
            q_color = STATE_COLORS["QUEUE"]
            cv2.fillPoly(overlay, [q_poly], q_color)
            cv2.polylines(frame, [q_poly], isClosed=True, color=q_color, thickness=2, lineType=cv2.LINE_AA)

            q_cx, q_cy = self.queue_region.get_centroid(w, h)
            q_label = f"QUEUE ({queue_count} Waiting)"
            (q_tw, q_th), _ = cv2.getTextSize(q_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            q_bx1 = max(5, q_cx - q_tw // 2 - 6)
            q_by1 = max(5, q_cy - q_th // 2 - 4)
            cv2.rectangle(frame, (q_bx1, q_by1), (q_bx1 + q_tw + 12, q_by1 + q_th + 8), (18, 22, 28), -1)
            cv2.rectangle(frame, (q_bx1, q_by1), (q_bx1 + q_tw + 12, q_by1 + q_th + 8), q_color, 1)
            cv2.putText(
                frame,
                q_label,
                (q_bx1 + 6, q_by1 + q_th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                q_color,
                1,
                cv2.LINE_AA,
            )

        # 3. Draw Entrance Zone
        if self.entrance_region is not None:
            e_poly = self.entrance_region.get_pixel_polygon(w, h)
            e_color = STATE_COLORS["ENTRANCE"]
            cv2.polylines(frame, [e_poly], isClosed=True, color=e_color, thickness=1, lineType=cv2.LINE_AA)
            e_cx, e_cy = self.entrance_region.get_centroid(w, h)
            cv2.putText(
                frame,
                "ENTRANCE",
                (e_cx - 35, e_cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                e_color,
                1,
                cv2.LINE_AA,
            )

        # Alpha blend overlay with frame
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame

