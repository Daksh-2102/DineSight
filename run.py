"""
DineSight AI - Master Integrated Application Runner
===================================================
Unifies all completed project phases into a single interactive demonstration:
- Phase 1: OpenCV Video Ingestion & Frame Sampling.
- Phase 2: YOLOv8 Object Detection & ByteTrack Persistent Tracking.
- Phase 3: Spatial ROI Mapping, Table-State Recognition (Vacant/Occupied/Dining),
           and Entrance Queue Monitoring.

Modes:
    1. 'full'   (Default): Complete end-to-end system (Phase 1 + 2 + 3).
    2. 'track'  : Video + YOLOv8 + ByteTrack IDs (Phase 2).
    3. 'detect' : Video + YOLOv8 Bounding Boxes only.
    4. 'raw'    : Baseline Video Ingestion without AI (Phase 1).

Usage:
    python run.py                          # Full system (Tables + Queue + Tracking)
    python run.py --mode track             # Tracking only (Phase 2)
    python run.py --mode raw               # Raw video stream only (Phase 1)
    python run.py --source 0               # Run on live laptop webcam
"""

import os
import sys
import time
import argparse
from typing import Dict, Optional, Union
import cv2
import numpy as np
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.core.video_reader import VideoReader
from backend.app.core.detector import YOLODetector
from backend.app.core.tracker import ByteTracker
from backend.app.core.roi_manager import ROIManager
from backend.app.core.state_engine import TableStateEngine
from backend.app.core.queue_monitor import QueueMonitor
from backend.app.core.entrance_monitor import EntranceMonitor


def draw_master_hud(
    frame: np.ndarray,
    mode: str,
    frame_idx: int,
    total_frames: int,
    pipeline_fps: float,
    inf_time_ms: float,
    device_name: str,
    count: int,
    total_unique: int,
    table_states: Optional[Dict[str, str]] = None,
    queue_metrics: Optional[Dict] = None,
    entrance_metrics: Optional[Dict] = None,
    frame_skip: int = 1,
    paused: bool = False,
) -> None:
    """
    Render a clean, responsive, multi-card glassmorphism HUD dashboard.
    Scales dynamically to fit any resolution without text overlapping or clipping.
    """
    h, w = frame.shape[:2]
    header_h = 68

    # --- Header Glass Background ---
    overlay_top = frame.copy()
    cv2.rectangle(overlay_top, (0, 0), (w, header_h), (14, 18, 24), -1)
    cv2.addWeighted(overlay_top, 0.90, frame, 0.10, 0, frame)

    # Accent bottom divider line
    cv2.line(frame, (0, header_h - 1), (w, header_h - 1), (35, 48, 64), 1, cv2.LINE_AA)

    mode_label = {
        "raw": "RAW INGESTION",
        "detect": "YOLO DETECTION",
        "track": "BYTETRACK",
        "full": "FULL ANALYTICS",
    }.get(mode.lower(), mode.upper())

    # --- 1. Left Section: Branding & System State ---
    cv2.putText(
        frame,
        "DINESIGHT AI",
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    # Mode pill badge
    (mw, mh), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    cv2.rectangle(frame, (142, 12), (142 + mw + 10, 12 + mh + 9), (28, 38, 52), -1)
    cv2.rectangle(frame, (142, 12), (142 + mw + 10, 12 + mh + 9), (0, 215, 255), 1)
    cv2.putText(
        frame,
        mode_label,
        (147, 12 + mh + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )

    # Subline: Device, FPS, Sampling, Frame
    inf_fps = (1000.0 / inf_time_ms) if inf_time_ms > 0 else 0.0
    sys_info = f"{device_name}  |  {pipeline_fps:.1f} FPS ({inf_time_ms:.1f}ms)  |  1/{frame_skip}  |  F:{frame_idx}/{total_frames}"
    cv2.putText(
        frame,
        sys_info,
        (14, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (165, 175, 190),
        1,
        cv2.LINE_AA,
    )

    # --- 2. Center Section: Dining Room Occupancy Card ---
    if mode == "full" and table_states is not None:
        total_tbl = len(table_states)
        dining_cnt = sum(1 for s in table_states.values() if s == "DINING")
        waiting_cnt = sum(1 for s in table_states.values() if s in ("WAITING_FOR_FOOD", "ESTIMATED_WAITING"))
        occupied_cnt = sum(1 for s in table_states.values() if s not in ("VACANT", "NONE"))
        vacant_cnt = total_tbl - occupied_cnt
        occ_pct = int((occupied_cnt / total_tbl) * 100) if total_tbl > 0 else 0

        card_cx = int(w * 0.40) if w >= 800 else int(w * 0.38)
        cv2.putText(
            frame,
            f"TABLES: {occupied_cnt}/{total_tbl} ({occ_pct}%) Seated",
            (card_cx, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        status_breakdown = f"{dining_cnt} Dining  |  {waiting_cnt} Waiting  |  {vacant_cnt} Vacant"
        cv2.putText(
            frame,
            status_breakdown,
            (card_cx, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (140, 220, 170),
            1,
            cv2.LINE_AA,
        )
    elif mode == "track":
        card_cx = int(w * 0.40)
        cv2.putText(frame, f"ACTIVE PERSONS: {count}", (card_cx, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"TOTAL OBSERVED: {total_unique}", (card_cx, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 220, 170), 1, cv2.LINE_AA)
    elif mode == "detect":
        card_cx = int(w * 0.40)
        cv2.putText(frame, f"DETECTED PERSONS: {count}", (card_cx, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "YOLOv8 Real-Time Inference", (card_cx, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 220, 170), 1, cv2.LINE_AA)

    # --- 3. Right Section: Service & Footfall Operations Card ---
    if mode == "full":
        right_x = max(int(w * 0.70), w - 265)
        q_len = (queue_metrics or {}).get("current_queue_length", 0)
        ent_in = (entrance_metrics or {}).get("total_entries", 0)
        ent_out = (entrance_metrics or {}).get("total_exits", 0)
        ent_net = (entrance_metrics or {}).get("net_occupants", 0)

        cv2.putText(
            frame,
            f"QUEUE: {q_len} waiting  |  PEOPLE: {count}",
            (right_x, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 215, 140),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FOOTFALL: IN {ent_in}  |  OUT {ent_out}  (Net {ent_net})",
            (right_x, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (200, 220, 255),
            1,
            cv2.LINE_AA,
        )

    # --- 4. Footer Bar ---
    footer_h = 26
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - footer_h), (w, h), (14, 18, 24), -1)
    cv2.addWeighted(overlay_bot, 0.90, frame, 0.10, 0, frame)
    cv2.line(frame, (0, h - footer_h), (w, h - footer_h), (35, 48, 64), 1, cv2.LINE_AA)

    # Status indicator dot + text
    status_str = "PAUSED [SPACE to resume]" if paused else "ACTIVE [SPACE to pause]"
    status_col = (0, 165, 255) if paused else (0, 255, 127)
    cv2.circle(frame, (20, h - footer_h // 2), 4, status_col, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        status_str,
        (30, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        status_col,
        1,
        cv2.LINE_AA,
    )

    tips = "[SPACE]: Pause  |  [r]: Reset  |  [+/-]: Sampling  |  [m]: Mode  |  [q]: Quit"
    (tip_w, _), _ = cv2.getTextSize(tips, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.putText(
        frame,
        tips,
        (max(int(w * 0.40), w - tip_w - 15), h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (160, 170, 180),
        1,
        cv2.LINE_AA,
    )


def main():
    load_dotenv()

    default_vid = os.getenv("DEFAULT_VIDEO_SOURCE", "data/sample_videos/restaurant_cctv_sample.mp4")
    if not os.path.exists(default_vid):
        default_vid = "data/sample_videos/pedestrians_surveillance.mp4"

    parser = argparse.ArgumentParser(
        description="DineSight AI - Master Project Runner"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["raw", "detect", "track", "full"],
        help="Operation mode: 'full' (Phase 3 tables+queue), 'track' (Phase 2), 'detect', 'raw'",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=default_vid,
        help="Path to video file, webcam index (e.g. 0), or RTSP URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/yolov8n.pt",
        help="Path to YOLO weights (.pt)",
    )
    parser.add_argument(
        "--rois",
        type=str,
        default="config/rois.json",
        help="Path to ROI configuration file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for person detection (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU threshold for NMS bounding box suppression (default: 0.50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (e.g. 640 or 800, default: 640)",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="config/bytetrack_dinesight.yaml" if os.path.exists("config/bytetrack_dinesight.yaml") else "bytetrack.yaml",
        help="Path to ByteTrack YAML configuration file",
    )
    parser.add_argument(
        "--assoc-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "bbox_overlap", "bottom_center", "center"],
        help="Spatial association mode for mapping persons to tables: 'hybrid' (default), 'bbox_overlap', 'bottom_center'",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.getenv("FRAME_WIDTH", 1280)),
        help="Target frame width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.getenv("FRAME_HEIGHT", 720)),
        help="Target frame height in pixels",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=int(os.getenv("FRAME_SKIP_INTERVAL", 1)),
        help="Frame skip interval (1 = all frames, 2 = every 2nd frame)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', 'cuda', or None (auto-detect)",
    )
    parser.add_argument(
        "--tableware",
        action="store_true",
        help="Enable COCO tableware detection (bottles, cups, bowls) for dining activity heuristics",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable automatic looping when video ends",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying a graphical window (useful for benchmarking)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process before exiting (0 = infinite)",
    )

    args = parser.parse_args()
    source_val: Union[str, int] = int(args.source) if args.source.isdigit() else args.source

    print("\n" + "=" * 70)
    print("             DINESIGHT AI - MASTER APPLICATION RUNNER")
    print("=" * 70)
    print(f"Active Mode  : {args.mode.upper()}")
    print(f"Video Source : {source_val}")

    # 1. Initialize Video Ingestion (Phase 1)
    try:
        reader = VideoReader(
            source=source_val,
            target_size=(args.width, args.height),
            frame_skip=args.skip,
            loop=not args.no_loop,
        )
    except FileNotFoundError as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Lazy Initialization for Phase 2 & 3 Components
    detector = None
    tracker = None
    roi_manager = None
    state_engine = None
    queue_monitor = None
    entrance_monitor = None
    device_name = "CPU"

    # Always initialize ROI and state engines for full mode
    if os.path.exists(args.rois):
        try:
            roi_manager = ROIManager(
                config_path=args.rois,
                point_mode="bottom_center",
                association_mode=args.assoc_mode,
            )
            table_ids = list(roi_manager.tables.keys())
            state_engine = TableStateEngine(table_ids=table_ids)
            queue_monitor = QueueMonitor()
            tripwire_line = None
            if roi_manager.entrance_region and roi_manager.entrance_region.tripwire_normalized:
                tripwire_line = roi_manager.entrance_region.tripwire_normalized
            entrance_monitor = EntranceMonitor(tripwire_line=tripwire_line)
        except Exception as e:
            print(f"[Warning] Could not initialize ROIManager: {e}")

    meta = reader.get_metadata()
    print(f"Resolution   : {meta['target_resolution']}\n")
    print("Keyboard Shortcuts:")
    print("  [SPACE] / [p] : Pause / Resume")
    print("  [r]           : Restart video and reset state engines")
    print("  [+] / [-]     : Dynamically adjust frame sampling")
    print("  [m]           : Live cycle modes (RAW -> DETECT -> TRACK -> FULL)")
    print("  [q] or [ESC]  : Quit\n")

    window_name = "DineSight AI - Master Application"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    delay_ms = max(1, int(1000.0 / (reader.source_fps or 30.0)))
    paused = False
    current_frame: Optional[np.ndarray] = None
    current_mode = args.mode
    current_idx = 0

    try:
        while True:
            now = time.time()

            if not paused:
                success, frame, current_idx = reader.read_frame()
                if not success or frame is None:
                    print("\n[DineSight AI] Video playback completed.")
                    break
                current_frame = frame
            else:
                time.sleep(0.03)

            if current_frame is not None:
                annotated = current_frame.copy()
                count = 0
                total_unique = 0
                inf_time_ms = 0.0
                table_states = {}
                queue_metrics = {}
                entrance_metrics = {}

                # Mode branching
                if current_mode == "detect":
                    if detector is None:
                        detector = YOLODetector(
                            model_path=args.model,
                            conf_threshold=args.conf,
                            iou_threshold=args.iou,
                            imgsz=args.imgsz,
                            target_classes=[0],
                            device=args.device,
                        )
                    dets = detector.detect(current_frame)
                    annotated = detector.draw_detections(annotated, dets)
                    count = len(dets)
                    inf_time_ms = detector.last_inference_time_ms
                    device_name = detector.device_name

                elif current_mode in ("track", "full"):
                    if tracker is None:
                        include_tw = (current_mode == "full" or args.tableware)
                        tracker = ByteTracker(
                            model_path=args.model,
                            conf_threshold=args.conf,
                            tracker_type=args.tracker_config,
                            imgsz=args.imgsz,
                            include_tableware=include_tw,
                            device=args.device,
                        )
                    trks = tracker.update(current_frame)
                    inf_time_ms = tracker.last_track_time_ms
                    device_name = tracker.device_name

                    if current_mode == "full" and roi_manager and state_engine and queue_monitor:
                        h, w = current_frame.shape[:2]
                        # Separate person tracks from tableware items
                        person_trks, tableware_items = ByteTracker.separate_tracks(trks)
                        count = len(person_trks)
                        total_unique = tracker.total_unique_tracks_seen

                        spatial_map = roi_manager.associate_tracks(person_trks, frame_width=w, frame_height=h)
                        table_tableware = roi_manager.associate_tableware(tableware_items, frame_width=w, frame_height=h)

                        table_states = state_engine.update(
                            spatial_map["tables"],
                            table_tableware=table_tableware,
                            now=now,
                        )
                        queue_metrics = queue_monitor.update(
                            queue_tracks=spatial_map["queue"],
                            counter_tracks=spatial_map.get("counter", []),
                            now=now,
                        )

                        if entrance_monitor:
                            e_poly = roi_manager.entrance_region.get_pixel_polygon(w, h) if roi_manager.entrance_region else None
                            entrance_metrics = entrance_monitor.update(
                                tracks=person_trks,
                                frame_width=w,
                                frame_height=h,
                                now=now,
                                entrance_polygon=e_poly,
                            )

                        # Draw ROIs
                        table_counts = {t_id: len(spatial_map["tables"].get(t_id, [])) for t_id in roi_manager.tables}
                        annotated = roi_manager.draw_rois(
                            annotated,
                            table_states=table_states,
                            table_occupant_counts=table_counts,
                            queue_count=queue_metrics.get("current_queue_length", 0),
                            counter_count=len(spatial_map.get("counter", [])),
                            entrance_metrics=entrance_metrics,
                            table_tableware=table_tableware,
                        )

                        if entrance_monitor:
                            annotated = entrance_monitor.draw_entrance_visuals(annotated)
                    else:
                        count = len(trks)
                        total_unique = tracker.total_unique_tracks_seen

                    # Draw person tracking boxes
                    annotated = tracker.draw_tracks(annotated, trks, show_center=True)

                # Draw Master HUD
                draw_master_hud(
                    frame=annotated,
                    mode=current_mode,
                    frame_idx=current_idx,
                    total_frames=reader.total_frames,
                    pipeline_fps=reader.current_fps,
                    inf_time_ms=inf_time_ms,
                    device_name=device_name,
                    count=count,
                    total_unique=total_unique,
                    table_states=table_states,
                    queue_metrics=queue_metrics,
                    entrance_metrics=entrance_metrics,
                    frame_skip=reader.frame_skip,
                    paused=paused,
                )

                if not args.headless:
                    cv2.imshow(window_name, annotated)

            if args.max_frames > 0 and current_idx >= args.max_frames:
                print(f"\n[DineSight AI] Reached target max frames ({args.max_frames}). Completed.")
                break

            # Keyboard controls (GUI mode only)
            if not args.headless:
                key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
                if key in (ord('q'), 27):
                    print("\n[DineSight AI] Exiting application.")
                    break
                elif key in (ord('p'), ord(' ')):
                    paused = not paused
                    print(f"[Playback] {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('r'):
                    reader.restart()
                    if tracker:
                        tracker.reset()
                    if state_engine:
                        state_engine.reset()
                    if queue_monitor:
                        queue_monitor.reset()
                    if entrance_monitor:
                        entrance_monitor.reset()
                    print("[Playback] Rewound and all state engines reset.")
                elif key in (ord('+'), ord('=')):
                    reader.frame_skip += 1
                    print(f"[Config] Frame skip: {reader.frame_skip}")
                elif key in (ord('-'), ord('_')):
                    reader.frame_skip = max(1, reader.frame_skip - 1)
                    print(f"[Config] Frame skip: {reader.frame_skip}")
                elif key == ord('m'):
                    modes = ["raw", "detect", "track", "full"]
                    current_mode = modes[(modes.index(current_mode) + 1) % len(modes)]
                    print(f"[Mode Switch] Active mode: {current_mode.upper()}")

    finally:
        reader.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
