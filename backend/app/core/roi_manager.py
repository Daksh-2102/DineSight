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
        tripwire_normalized: Optional[List[List[float]]] = None,
    ):
        """
        Args:
            region_id: Unique string identifier (e.g. 'table_1', 'queue_zone').
            name: Human-readable display label (e.g. 'Table 1 (2-Seater)').
            polygon_normalized: List of [x, y] coordinates in [0.0, 1.0].
            region_type: 'table', 'queue', or 'entrance'.
            max_capacity: Seating capacity (for tables).
            tripwire_normalized: Optional [[x1, y1], [x2, y2]] boundary line coordinates.
        """
        self.region_id = str(region_id)
        self.name = str(name)
        self.region_type = region_type.lower()
        self.max_capacity = int(max_capacity)
        self.tripwire_normalized = tripwire_normalized

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

    def compute_bbox_overlap_ratio(
        self,
        bbox: List[int],
        frame_width: int,
        frame_height: int,
    ) -> float:
        """
        Compute the intersection area ratio of a bounding box [x1, y1, x2, y2]
        relative to the bounding box's total area.
        Returns float in [0.0, 1.0].
        """
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        bbox_area = float(bw * bh)

        poly = self.get_pixel_polygon(frame_width, frame_height)
        # Shift polygon into local bounding box coordinates
        shifted_poly = poly.reshape((-1, 2)) - [x1, y1]

        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_poly], 255)
        overlap_area = float(cv2.countNonZero(mask))

        return overlap_area / bbox_area

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
        association_mode: str = "hybrid",
        overlap_threshold: float = 0.25,
    ):
        """
        Args:
            config_path: Path to JSON configuration file.
            point_mode: How to extract representative point from a person bounding box.
                        'bottom_center': ((x1+x2)//2, y2) -> Best for floor/table projection.
                        'center': ((x1+x2)//2, (y1+y2)//2) -> Center of mass.
            association_mode: Spatial association strategy:
                              'hybrid': Combines bounding-box overlap ratio (>=25%) with
                                        centroid and bottom-center fallbacks. (Recommended)
                              'bbox_overlap': Pure area overlap thresholding.
                              'point': Legacy single-point containment.
            overlap_threshold: Minimum bounding-box intersection ratio (default 0.25 or 25%).
        """
        self.config_path = config_path
        self.point_mode = point_mode.lower()
        if self.point_mode not in ("bottom_center", "center"):
            self.point_mode = "bottom_center"

        self.association_mode = association_mode.lower()
        self.overlap_threshold = float(overlap_threshold)

        self.tables: Dict[str, RegionOfInterest] = {}
        self.queue_region: Optional[RegionOfInterest] = None
        self.entrance_region: Optional[RegionOfInterest] = None
        self.counter_region: Optional[RegionOfInterest] = None

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
        self.counter_region = None

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
                tripwire_normalized=e_data.get("tripwire"),
            )

        # 4. Parse Service Counter
        if "counter" in data and data["counter"]:
            c_data = data["counter"]
            self.counter_region = RegionOfInterest(
                region_id=c_data.get("id", "counter_zone"),
                name=c_data.get("name", "Service Counter"),
                polygon_normalized=c_data["polygon"],
                region_type="counter",
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

    def find_table_for_bbox(
        self,
        bbox: List[int],
        frame_width: int,
        frame_height: int,
    ) -> Optional[str]:
        """
        Find which table best associates with a bounding box [x1, y1, x2, y2].
        Uses intersection overlap area ratio and multi-anchor verification.
        """
        best_table = None
        best_overlap = 0.0

        # Primary metric: maximum area overlap ratio
        for t_id, table in self.tables.items():
            ov = table.compute_bbox_overlap_ratio(bbox, frame_width, frame_height)
            if ov >= self.overlap_threshold and ov > best_overlap:
                best_overlap = ov
                best_table = t_id

        # Fallback to multi-anchor point containment if overlap below threshold
        if best_table is None and self.association_mode == "hybrid":
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroid = (cx, cy)
            bottom_center = (cx, int(y2))
            lower_torso = (cx, int(y1 + 0.75 * (y2 - y1)))

            for pt in (lower_torso, centroid, bottom_center):
                matched = self.find_table_for_point(pt, frame_width, frame_height)
                if matched is not None:
                    return matched

        return best_table

    def is_in_counter(self, point: Tuple[int, int], frame_width: int, frame_height: int) -> bool:
        """Check if a point is inside the service counter zone."""
        if self.counter_region is None:
            return False
        return self.counter_region.contains_point(point, frame_width, frame_height)

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

    def get_entrance_tripwire(
        self,
        frame_width: int,
        frame_height: int,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Return the entrance boundary tripwire line in absolute pixel coordinates.
        Uses configured tripwire if available, otherwise computes the bottom edge
        of the entrance polygon.
        """
        if self.entrance_region is None:
            return None

        if self.entrance_region.tripwire_normalized is not None:
            p1_norm, p2_norm = self.entrance_region.tripwire_normalized
            p1 = (int(round(p1_norm[0] * frame_width)), int(round(p1_norm[1] * frame_height)))
            p2 = (int(round(p2_norm[0] * frame_width)), int(round(p2_norm[1] * frame_height)))
            return p1, p2

        # Automatically derive bottom horizontal edge of entrance polygon
        poly = self.entrance_region.polygon_normalized
        sorted_by_y = sorted(poly, key=lambda pt: pt[1], reverse=True)
        if len(sorted_by_y) >= 2:
            pt_a, pt_b = sorted_by_y[0], sorted_by_y[1]
            p1 = (int(round(min(pt_a[0], pt_b[0]) * frame_width)), int(round(pt_a[1] * frame_height)))
            p2 = (int(round(max(pt_a[0], pt_b[0]) * frame_width)), int(round(pt_b[1] * frame_height)))
            return p1, p2

        return None

    def associate_tracks(
        self,
        tracks: List[Dict],
        frame_width: int,
        frame_height: int,
    ) -> Dict:
        """
        Perform spatial mapping: takes all current tracked people and associates
        them with tables, queue, counter, or unassigned floor area.
        """
        mapping: Dict[str, Union[Dict[str, List[Dict]], List[Dict]]] = {
            "tables": {t_id: [] for t_id in self.tables},
            "queue": [],
            "counter": [],
            "entrance": [],
            "unassigned": [],
        }

        for trk in tracks:
            # Skip non-person tracks (tableware items) so they don't count as human patrons
            if trk.get("class_id", 0) != 0 and trk.get("class_name", "person") != "person":
                continue

            bbox = trk["bbox"]
            pt = self.get_representative_point(bbox)
            trk_with_pt = {**trk, "representative_point": pt}

            # 1. Check Table ROIs using configured association mode
            if self.association_mode in ("hybrid", "bbox_overlap"):
                matched_table = self.find_table_for_bbox(bbox, frame_width, frame_height)
            else:
                matched_table = self.find_table_for_point(pt, frame_width, frame_height)

            if matched_table is not None:
                mapping["tables"][matched_table].append(trk_with_pt)
                continue

            # 2. Check Counter ROI (point or overlap)
            in_counter = self.is_in_counter(pt, frame_width, frame_height)
            if not in_counter and self.counter_region and self.association_mode in ("hybrid", "bbox_overlap"):
                in_counter = self.counter_region.compute_bbox_overlap_ratio(bbox, frame_width, frame_height) >= self.overlap_threshold

            if in_counter:
                mapping["counter"].append(trk_with_pt)
                # Counter occupants also count toward queue analysis
                mapping["queue"].append(trk_with_pt)
                continue

            # 3. Check Queue ROI (point or overlap)
            in_queue = self.is_in_queue(pt, frame_width, frame_height)
            if not in_queue and self.queue_region and self.association_mode in ("hybrid", "bbox_overlap"):
                in_queue = self.queue_region.compute_bbox_overlap_ratio(bbox, frame_width, frame_height) >= self.overlap_threshold

            if in_queue:
                mapping["queue"].append(trk_with_pt)
                continue

            # 4. Check Entrance ROI (point or overlap)
            in_entrance = self.is_in_entrance(pt, frame_width, frame_height)
            if not in_entrance and self.entrance_region and self.association_mode in ("hybrid", "bbox_overlap"):
                in_entrance = self.entrance_region.compute_bbox_overlap_ratio(bbox, frame_width, frame_height) >= self.overlap_threshold

            if in_entrance:
                mapping["entrance"].append(trk_with_pt)
                continue

            # 5. Otherwise Unassigned / Walking in Aisles
            mapping["unassigned"].append(trk_with_pt)

        return mapping

    def associate_tableware(
        self,
        tableware_items: List[Dict],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, List[Dict]]:
        """
        Associate detected tableware items (bottles, cups, bowls, cutlery) with table ROIs.
        Returns a dictionary mapping table_id -> list of tableware items located on that table.
        """
        table_tableware: Dict[str, List[Dict]] = {t_id: [] for t_id in self.tables}
        for item in tableware_items:
            bbox = item["bbox"]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            pt = (cx, cy)

            matched_table = self.find_table_for_point(pt, frame_width, frame_height)
            if matched_table is None and self.association_mode in ("hybrid", "bbox_overlap"):
                matched_table = self.find_table_for_bbox(bbox, frame_width, frame_height)

            if matched_table is not None:
                item_with_pt = {**item, "representative_point": pt}
                table_tableware[matched_table].append(item_with_pt)

        return table_tableware

    def draw_rois(
        self,
        frame: np.ndarray,
        table_states: Optional[Dict[str, str]] = None,
        table_occupant_counts: Optional[Dict[str, int]] = None,
        queue_count: int = 0,
        counter_count: int = 0,
        entrance_metrics: Optional[Dict] = None,
        table_tableware: Optional[Dict[str, List[Dict]]] = None,
        alpha: float = 0.12,
    ) -> np.ndarray:
        """
        Render elegant, non-intrusive colored polygons, crisp perimeter outlines,
        and top-anchored status badges for all active restaurant ROIs.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Dining Tables
        for t_id, table in self.tables.items():
            poly = table.get_pixel_polygon(w, h)
            state = (table_states or {}).get(t_id, "VACANT").upper()
            color = STATE_COLORS.get(state, STATE_COLORS["VACANT"])

            # Delicate semi-transparent polygon fill
            cv2.fillPoly(overlay, [poly], color)

            # Solid perimeter border
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Anchor badge to the top perimeter border of the table
            pts = poly.reshape((-1, 2))
            x_min, y_min = np.min(pts, axis=0)
            x_max, _ = np.max(pts, axis=0)
            top_cx = int((x_min + x_max) / 2)

            # Compact, readable label (e.g. 'Table 1: DINING (2p) [Food]')
            clean_name = table.name.split("(")[0].strip()
            occupants = (table_occupant_counts or {}).get(t_id, 0)
            tw_items = (table_tableware or {}).get(t_id, [])
            tw_count = len(tw_items)

            state_display = "WAITING" if "WAITING" in state else state
            label = f"{clean_name}: {state_display}"
            if occupants > 0:
                label += f" ({occupants}p)"
            if tw_count > 0:
                label += " [Food]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            thickness = 1
            (tw_sz, th_sz), _ = cv2.getTextSize(label, font, font_scale, thickness)

            badge_x1 = max(5, top_cx - tw_sz // 2 - 5)
            badge_y1 = max(72, y_min - th_sz - 6)
            badge_x2 = badge_x1 + tw_sz + 10
            badge_y2 = badge_y1 + th_sz + 6

            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 22, 28), -1)
            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)
            cv2.putText(
                frame,
                label,
                (badge_x1 + 5, badge_y2 - 4),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            # Draw subtle tableware indicators on table surface
            for tw_item in tw_items:
                bx1, by1, bx2, by2 = tw_item["bbox"]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 140, 0), 1)
                tw_name = tw_item.get("class_name", "item")
                cv2.putText(
                    frame,
                    tw_name.capitalize(),
                    (bx1, max(10, by1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 180, 50),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Service Counter Zone
        if self.counter_region is not None:
            c_poly = self.counter_region.get_pixel_polygon(w, h)
            c_color = (210, 180, 50)  # Slate Cyan
            cv2.fillPoly(overlay, [c_poly], c_color)
            cv2.polylines(frame, [c_poly], isClosed=True, color=c_color, thickness=1, lineType=cv2.LINE_AA)

            c_pts = c_poly.reshape((-1, 2))
            cx_min, cy_min = np.min(c_pts, axis=0)
            cx_max = np.max(c_pts[:, 0])
            c_label = f"COUNTER ({counter_count} Active)" if counter_count > 0 else "SERVICE COUNTER"

            (ctw, cth), _ = cv2.getTextSize(c_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            cbx = int((cx_min + cx_max) / 2) - ctw // 2 - 5
            cby = max(72, cy_min - cth - 6)
            cv2.rectangle(frame, (cbx, cby), (cbx + ctw + 10, cby + cth + 6), (18, 22, 28), -1)
            cv2.rectangle(frame, (cbx, cby), (cbx + ctw + 10, cby + cth + 6), c_color, 1)
            cv2.putText(
                frame,
                c_label,
                (cbx + 5, cby + cth + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                c_color,
                1,
                cv2.LINE_AA,
            )

        # 3. Queue Zone
        if self.queue_region is not None:
            q_poly = self.queue_region.get_pixel_polygon(w, h)
            q_color = STATE_COLORS["QUEUE"]
            cv2.fillPoly(overlay, [q_poly], q_color)
            cv2.polylines(frame, [q_poly], isClosed=True, color=q_color, thickness=1, lineType=cv2.LINE_AA)

            q_pts = q_poly.reshape((-1, 2))
            qx_min, qy_min = np.min(q_pts, axis=0)
            qx_max = np.max(q_pts[:, 0])
            q_label = f"QUEUE ({queue_count} Waiting)"

            (q_tw, q_th), _ = cv2.getTextSize(q_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            q_bx1 = int((qx_min + qx_max) / 2) - q_tw // 2 - 5
            q_by1 = max(72, qy_min - q_th - 6)
            cv2.rectangle(frame, (q_bx1, q_by1), (q_bx1 + q_tw + 10, q_by1 + q_th + 6), (18, 22, 28), -1)
            cv2.rectangle(frame, (q_bx1, q_by1), (q_bx1 + q_tw + 10, q_by1 + q_th + 6), q_color, 1)
            cv2.putText(
                frame,
                q_label,
                (q_bx1 + 5, q_by1 + q_th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                q_color,
                1,
                cv2.LINE_AA,
            )

        # 4. Entrance Zone (Subtle outline, no duplicate footfall numbers)
        if self.entrance_region is not None:
            e_poly = self.entrance_region.get_pixel_polygon(w, h)
            e_color = STATE_COLORS["ENTRANCE"]
            cv2.polylines(frame, [e_poly], isClosed=True, color=e_color, thickness=1, lineType=cv2.LINE_AA)

            e_pts = e_poly.reshape((-1, 2))
            ex_min, ey_min = np.min(e_pts, axis=0)
            ex_max = np.max(e_pts[:, 0])
            e_label = "ENTRANCE ZONE"
            (e_tw, e_th), _ = cv2.getTextSize(e_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            ebx1 = int((ex_min + ex_max) / 2) - e_tw // 2 - 5
            eby1 = max(72, ey_min - e_th - 6)
            cv2.rectangle(frame, (ebx1, eby1), (ebx1 + e_tw + 10, eby1 + e_th + 6), (18, 22, 28), -1)
            cv2.rectangle(frame, (ebx1, eby1), (ebx1 + e_tw + 10, eby1 + e_th + 6), e_color, 1)
            cv2.putText(
                frame,
                e_label,
                (ebx1 + 5, eby1 + e_th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                e_color,
                1,
                cv2.LINE_AA,
            )

        # Alpha blend subtle overlay with frame
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame

