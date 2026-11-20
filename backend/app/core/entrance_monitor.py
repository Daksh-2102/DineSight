"""
DineSight AI - Entrance Flow & Customer Footfall Monitoring Engine (Module 3D)
==============================================================================
Monitors restaurant customer footfall across the designated entrance portal.

Key Responsibilities:
- Tracks directional movement vectors (Entry vs. Exit) across a virtual boundary / tripwire.
- Distinguishes customers walking into the dining room from customers exiting the premises.
- Implements spatial hysteresis and debounce timers to eliminate double-counting from
  lingering, hesitating, or swaying patrons near the doorway.
- Computes real-time footfall analytics:
  - Cumulative Customer Entries (Footfall In)
  - Cumulative Customer Exits (Footfall Out)
  - Net In-Building Occupancy (In - Out)
  - Real-time entrance zone occupancy
  - Rolling entry arrival rate (flow rate per minute)
- Maintains rolling event logs with timestamps and direction vectors.
- Ephemeral in-memory telemetry: No facial profiles, biometrics, or permanent identifiers.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class FootfallEvent:
    """
    Represents an individual entry or exit crossing event.
    """

    def __init__(
        self,
        track_id: int,
        direction: str,  # "ENTRY" or "EXIT"
        timestamp: float,
        position: Tuple[int, int],
        velocity_vector: Tuple[float, float],
    ):
        self.track_id = int(track_id)
        self.direction = str(direction)
        self.timestamp = float(timestamp)
        self.position = position
        self.velocity_vector = velocity_vector

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "direction": self.direction,
            "timestamp": self.timestamp,
            "position": list(self.position),
            "velocity": list(self.velocity_vector),
        }


class EntranceTrackTrajectory:
    """
    Tracks trajectory history and directional state for an individual tracked person.
    """

    def __init__(self, track_id: int, initial_pos: Tuple[int, int], timestamp: float):
        self.track_id = int(track_id)
        self.positions: List[Tuple[float, Tuple[int, int]]] = [(timestamp, initial_pos)]
        self.first_seen_time = float(timestamp)
        self.last_seen_time = float(timestamp)
        self.last_counted_direction: Optional[str] = None
        self.last_counted_time: Optional[float] = None
        # Confirmed spatial side relative to boundary: "EXTERIOR", "INTERIOR", or "UNKNOWN"
        self.confirmed_side: str = "UNKNOWN"

    def add_position(self, pos: Tuple[int, int], timestamp: float, max_history: int = 30) -> None:
        self.positions.append((float(timestamp), pos))
        self.last_seen_time = float(timestamp)
        if len(self.positions) > max_history:
            self.positions.pop(0)

    def get_displacement_vector(self) -> Tuple[float, float]:
        """Compute net displacement vector (dx, dy) from oldest to newest position."""
        if len(self.positions) < 2:
            return (0.0, 0.0)
        old_pt = self.positions[0][1]
        new_pt = self.positions[-1][1]
        return (float(new_pt[0] - old_pt[0]), float(new_pt[1] - old_pt[1]))

    def get_latest_position(self) -> Tuple[int, int]:
        return self.positions[-1][1]


class EntranceMonitor:
    """
    Directional Customer Footfall and Entrance Flow Monitor.
    Uses calibrated boundary tripwires with spatial hysteresis and debouncing.
    """

    def __init__(
        self,
        tripwire_line: Optional[Union[List[List[float]], Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
        hysteresis_band_px: float = 15.0,
        lateral_margin: float = 0.15,
        debounce_cooldown_sec: float = 4.0,
        loss_grace_sec: float = 3.0,
        inward_normal: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            tripwire_line: Optional normalized or pixel coordinates for counting line [[x1, y1], [x2, y2]].
            hysteresis_band_px: Spatial hysteresis buffer (+/- px) to prevent oscillation.
            lateral_margin: Permissible tolerance beyond line endpoints (0.15 = 15% overshoot allowed).
            debounce_cooldown_sec: Cooldown before same track ID can trigger same crossing again.
            loss_grace_sec: Seconds before purging missing track history from memory.
            inward_normal: Vector indicating interior direction. Defaults to (0, 1) [downward into room].
        """
        self.raw_tripwire = tripwire_line
        self.hysteresis_band_px = float(hysteresis_band_px)
        self.lateral_margin = float(lateral_margin)
        self.debounce_cooldown_sec = float(debounce_cooldown_sec)
        self.loss_grace_sec = float(loss_grace_sec)
        self.inward_normal = inward_normal or (0.0, 1.0)

        # Cumulative Metrics
        self.total_entries: int = 0
        self.total_exits: int = 0

        # Ephemeral active state
        self.active_tracks: Dict[int, EntranceTrackTrajectory] = {}
        self.completed_events: List[FootfallEvent] = []
        self.recent_events_log: List[Dict] = []

        # Internal cached pixel geometry
        self._cached_tripwire_pts: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self._cached_dims: Tuple[int, int] = (0, 0)

    @property
    def net_occupancy(self) -> int:
        """Net patrons currently inside premises based on footfall (floor at 0)."""
        return max(0, self.total_entries - self.total_exits)

    def reset(self) -> None:
        """Reset all footfall metrics, track trajectories, and event histories."""
        self.total_entries = 0
        self.total_exits = 0
        self.active_tracks.clear()
        self.completed_events.clear()
        self.recent_events_log.clear()

    def set_tripwire(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        """Dynamically set or recalibrate tripwire line."""
        self.raw_tripwire = [list(p1), list(p2)]
        self._cached_tripwire_pts = None

    def get_pixel_tripwire(self, frame_width: int, frame_height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Resolve tripwire endpoints in absolute pixel coordinates.
        Supports both normalized [0..1] and absolute pixel definitions.
        """
        if self._cached_tripwire_pts is not None and self._cached_dims == (frame_width, frame_height):
            return self._cached_tripwire_pts

        if self.raw_tripwire is not None and len(self.raw_tripwire) == 2:
            p1_raw, p2_raw = self.raw_tripwire
            # Check if normalized coordinates
            if max(p1_raw[0], p1_raw[1], p2_raw[0], p2_raw[1]) <= 1.0:
                p1 = (int(round(p1_raw[0] * frame_width)), int(round(p1_raw[1] * frame_height)))
                p2 = (int(round(p2_raw[0] * frame_width)), int(round(p2_raw[1] * frame_height)))
            else:
                p1 = (int(round(p1_raw[0])), int(round(p1_raw[1])))
                p2 = (int(round(p2_raw[0])), int(round(p2_raw[1])))
        else:
            # Fallback default: horizontal line across top-left entrance (x: 2%..35%, y: 32%)
            p1 = (int(round(0.02 * frame_width)), int(round(0.32 * frame_height)))
            p2 = (int(round(0.35 * frame_width)), int(round(0.32 * frame_height)))

        # Ensure ordered left-to-right
        if p1[0] > p2[0]:
            p1, p2 = p2, p1

        self._cached_tripwire_pts = (p1, p2)
        self._cached_dims = (frame_width, frame_height)
        return p1, p2

    def _compute_signed_distance_and_proj(
        self,
        point: Tuple[int, int],
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        Compute signed perpendicular distance from point to directed line AB,
        and projection factor along segment AB.

        Returns:
            (signed_distance_px, proj_ratio)
            - signed_distance: positive if on interior side, negative if on exterior side.
            - proj_ratio: 0.0 at A, 1.0 at B.
        """
        ax, ay = a
        bx, by = b
        px, py = point

        ab_x = float(bx - ax)
        ab_y = float(by - ay)
        ap_x = float(px - ax)
        ap_y = float(py - ay)

        length = np.hypot(ab_x, ab_y)
        if length < 1e-5:
            return 0.0, 0.0

        # Signed distance via 2D cross product: (AB_x * AP_y - AB_y * AP_x) / length
        cross = (ab_x * ap_y - ab_y * ap_x) / length

        # Inward normal check: ensure positive distance corresponds to inward direction
        # If line runs roughly left-to-right (ab_x > 0), cross > 0 means py > line_y (downward / interior).
        # We can align with configured self.inward_normal:
        normal_x = -ab_y / length
        normal_y = ab_x / length
        dot_normal = normal_x * self.inward_normal[0] + normal_y * self.inward_normal[1]
        if dot_normal < 0:
            cross = -cross

        # Projection factor along segment
        proj = (ap_x * ab_x + ap_y * ab_y) / (length * length)
        return cross, proj

    def update(
        self,
        tracks: List[Dict],
        frame_width: int,
        frame_height: int,
        now: Optional[float] = None,
        entrance_polygon: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Process tracks in the current frame, update trajectories, and evaluate
        directional boundary crossings with debounce hysteresis.

        Args:
            tracks: List of track dicts from ByteTracker.
            frame_width: Width of video frame.
            frame_height: Height of video frame.
            now: Current timestamp (seconds). Uses time.time() if None.
            entrance_polygon: Optional pixel polygon for entrance ROI containment.

        Returns:
            Dictionary with comprehensive footfall analytics.
        """
        if now is None:
            now = time.time()

        p1, p2 = self.get_pixel_tripwire(frame_width, frame_height)
        seen_track_ids = set()
        new_entries_this_frame = 0
        new_exits_this_frame = 0
        count_in_entrance_zone = 0

        # 1. Update positions & evaluate boundary state transitions
        for trk in tracks:
            tid = int(trk["track_id"])
            seen_track_ids.add(tid)

            # Use bottom-center point (foot contact plane) for ground plane accuracy
            bbox = trk["bbox"]
            px = int((bbox[0] + bbox[2]) / 2)
            py = int(bbox[3])
            pt = (px, py)

            # Optional entrance polygon check
            if entrance_polygon is not None:
                if cv2.pointPolygonTest(entrance_polygon, (float(px), float(py)), False) >= 0:
                    count_in_entrance_zone += 1

            if tid not in self.active_tracks:
                self.active_tracks[tid] = EntranceTrackTrajectory(tid, pt, now)
            else:
                self.active_tracks[tid].add_position(pt, now)

            traj = self.active_tracks[tid]

            # Compute signed distance and lateral projection
            signed_dist, proj = self._compute_signed_distance_and_proj(pt, p1, p2)

            # Check lateral containment: within line span plus margin
            is_laterally_aligned = (-self.lateral_margin <= proj <= 1.0 + self.lateral_margin)

            # Evaluate Hysteresis State Machine
            if is_laterally_aligned:
                current_side = "TRANSITION"
                if signed_dist > self.hysteresis_band_px:
                    current_side = "INTERIOR"
                elif signed_dist < -self.hysteresis_band_px:
                    current_side = "EXTERIOR"

                # If first time in zone, initialize side
                if traj.confirmed_side == "UNKNOWN":
                    if current_side in ("INTERIOR", "EXTERIOR"):
                        traj.confirmed_side = current_side

                # Detect Transitions
                elif traj.confirmed_side == "EXTERIOR" and current_side == "INTERIOR":
                    # Transition: EXTERIOR -> INTERIOR = ENTRY
                    can_count = True
                    if traj.last_counted_direction == "ENTRY" and traj.last_counted_time is not None:
                        if (now - traj.last_counted_time) < self.debounce_cooldown_sec:
                            can_count = False

                    if can_count:
                        self.total_entries += 1
                        new_entries_this_frame += 1
                        traj.last_counted_direction = "ENTRY"
                        traj.last_counted_time = now
                        traj.confirmed_side = "INTERIOR"

                        disp = traj.get_displacement_vector()
                        evt = FootfallEvent(tid, "ENTRY", now, pt, disp)
                        self.completed_events.append(evt)
                        self.recent_events_log.append(evt.to_dict())
                    else:
                        traj.confirmed_side = "INTERIOR"

                elif traj.confirmed_side == "INTERIOR" and current_side == "EXTERIOR":
                    # Transition: INTERIOR -> EXTERIOR = EXIT
                    can_count = True
                    if traj.last_counted_direction == "EXIT" and traj.last_counted_time is not None:
                        if (now - traj.last_counted_time) < self.debounce_cooldown_sec:
                            can_count = False

                    if can_count:
                        self.total_exits += 1
                        new_exits_this_frame += 1
                        traj.last_counted_direction = "EXIT"
                        traj.last_counted_time = now
                        traj.confirmed_side = "EXTERIOR"

                        disp = traj.get_displacement_vector()
                        evt = FootfallEvent(tid, "EXIT", now, pt, disp)
                        self.completed_events.append(evt)
                        self.recent_events_log.append(evt.to_dict())
                    else:
                        traj.confirmed_side = "EXTERIOR"

        # 2. Ephemeral Garbage Collection: Purge inactive tracks after loss grace period
        stale_ids = [
            tid
            for tid, traj in self.active_tracks.items()
            if (now - traj.last_seen_time) > self.loss_grace_sec
        ]
        for tid in stale_ids:
            del self.active_tracks[tid]

        # 3. Rolling arrival rate (entries in last 60 seconds)
        cutoff_60s = now - 60.0
        entries_last_min = sum(1 for e in self.completed_events if e.direction == "ENTRY" and e.timestamp >= cutoff_60s)
        flow_rate_per_min = float(entries_last_min)

        return {
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "net_occupants": self.net_occupancy,
            "current_in_entrance": count_in_entrance_zone,
            "new_entries": new_entries_this_frame,
            "new_exits": new_exits_this_frame,
            "flow_rate_per_min": flow_rate_per_min,
            "active_entrance_tracks": len(self.active_tracks),
            "recent_events": self.recent_events_log[-10:],
        }

    def draw_entrance_visuals(
        self,
        frame: np.ndarray,
        show_tripwire: bool = True,
        show_badge: bool = True,
        show_trails: bool = True,
    ) -> np.ndarray:
        """
        Draw calibrated entrance tripwire, directional indicators, and footfall telemetry badge.
        """
        h, w = frame.shape[:2]
        p1, p2 = self.get_pixel_tripwire(w, h)

        # 1. Draw glowing Tripwire Boundary Line
        if show_tripwire:
            # Subtle glow line underneath
            glow_color = (255, 140, 0)  # Bright Orange/Cyan in BGR
            cv2.line(frame, p1, p2, glow_color, 4, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (255, 230, 180), 2, cv2.LINE_AA)

            # Draw directional arrows at line midpoint
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)

            # Inward entry arrow (downward)
            cv2.arrowedLine(
                frame,
                (mid_x - 30, mid_y - 14),
                (mid_x - 30, mid_y + 14),
                (0, 255, 127),  # Green
                2,
                tipLength=0.35,
            )
            cv2.putText(
                frame,
                "IN",
                (mid_x - 48, mid_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 127),
                1,
                cv2.LINE_AA,
            )

            # Outward exit arrow (upward)
            cv2.arrowedLine(
                frame,
                (mid_x + 30, mid_y + 14),
                (mid_x + 30, mid_y - 14),
                (0, 140, 255),  # Amber
                2,
                tipLength=0.35,
            )
            cv2.putText(
                frame,
                "OUT",
                (mid_x + 38, mid_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 140, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Trajectory Motion Trails for active visitors near entrance
        if show_trails:
            for tid, traj in self.active_tracks.items():
                if len(traj.positions) >= 2:
                    pts = [p for _, p in traj.positions]
                    for k in range(1, len(pts)):
                        pt_a = pts[k - 1]
                        pt_b = pts[k]
                        alpha_trail = int(80 + (175 * k / len(pts)))
                        cv2.line(frame, pt_a, pt_b, (0, alpha_trail, 255), 1, cv2.LINE_AA)

        # 3. Draw Telemetry Badge
        if show_badge:
            label = f"DOORWAY TRIPWIRE [In: {self.total_entries} | Out: {self.total_exits}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            mid_x = int((p1[0] + p2[0]) / 2)
            bx1 = max(5, mid_x - tw // 2 - 6)
            by1 = max(72, p1[1] - th - 18)
            bx2 = bx1 + tw + 12
            by2 = by1 + th + 6

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (18, 22, 28), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 140, 0), 1)
            cv2.putText(
                frame,
                label,
                (bx1 + 6, by2 - 4),
                font,
                font_scale,
                (255, 220, 150),
                thickness,
                cv2.LINE_AA,
            )

        return frame
