"""
DineSight AI - Phase 3 Runner Script
====================================
Interactive launcher for ROI Mapping, Table-State Recognition, and Queue Monitoring.

Integrated Pipeline:
    Recorded Video
    -> OpenCV VideoReader
    -> YOLOv8 Person & Tableware Detection
    -> ByteTrack Multi-Object Tracking
    -> ROIManager Spatial Association
    -> TableStateEngine (Vacant / Occupied / Waiting / Dining)
    -> QueueMonitor (Queue Length & Elapsed Waiting Times)
    -> Professional CCTV Telemetry HUD & Color-Coded Spatial Polygons

Usage:
    python run_phase3.py
    python run_phase3.py --source 0                         # Webcam
    python run_phase3.py --source path/to/video.mp4         # Custom video
    python run_phase3.py --rois config/rois.json            # Custom ROI config
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.core.video_reader import VideoReader
from backend.app.core.detector import YOLODetector
from backend.app.core.tracker import ByteTracker
from backend.app.core.roi_manager import ROIManager, STATE_COLORS
from backend.app.core.state_engine import TableStateEngine, TableState
from backend.app.core.queue_monitor import QueueMonitor


def draw_phase3_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    pipeline_fps: float,
    inf_time_ms: float,
    device_name: str,
    table_states: Dict[str, str],
    queue_metrics: Dict,
    total_tracked_people: int,
    frame_skip: int,
    paused: bool,
) -> None:
    """Render a comprehensive telemetry HUD for Phase 3."""
    h, w = frame.shape[:2]

    # --- Header Bar ---
    header_h = 65
    overlay_top = frame.copy()
    cv2.rectangle(overlay_top, (0, 0), (w, header_h), (16, 20, 26), -1)
    cv2.addWeighted(overlay_top, 0.88, frame, 0.12, 0, frame)

    # Title
    cv2.putText(
        frame,
        "DineSight AI | Smart Table & Queue Analytics",
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),  # Gold
        2,
        cv2.LINE_AA,
    )

    # Compute & Device Info
    dev_col = (0, 255, 127) if "GPU" in device_name else (200, 200, 200)
    dev_text = f"Device: {device_name}  |  Sampling: 1/{frame_skip}"
    cv2.putText(frame, dev_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dev_col, 1, cv2.LINE_AA)

    # Performance & Table Summary
    total_tables = len(table_states)
    occupied_count = sum(1 for s in table_states.values() if s != "VACANT")
    dining_count = sum(1 for s in table_states.values() if s == "DINING")
    waiting_count = sum(1 for s in table_states.values() if s == "WAITING_FOR_FOOD")
    vacant_count = total_tables - occupied_count
    occupancy_pct = int((occupied_count / total_tables) * 100) if total_tables > 0 else 0

    inf_fps = (1000.0 / inf_time_ms) if inf_time_ms > 0 else 0.0
    line1 = f"Inference: {inf_time_ms:4.1f}ms ({inf_fps:3.0f} FPS)  |  Pipeline: {pipeline_fps:4.1f} FPS  |  Tracked: {total_tracked_people}"
    line2 = f"Tables: {occupied_count}/{total_tables} ({occupancy_pct}%) [Vacant:{vacant_count}, Wait:{waiting_count}, Dine:{dining_count}]"

    cv2.putText(
        frame,
        line1,
        (w - 680 if w >= 1050 else 15, 26 if w >= 1050 else 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        line2,
        (w - 680 if w >= 1050 else 15, 50 if w >= 1050 else 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 215, 240),
        1,
        cv2.LINE_AA,
    )

    # --- Live Queue Widget (Top Right / Sub-bar) ---
    q_len = queue_metrics.get("current_queue_length", 0)
    q_wait = queue_metrics.get("longest_current_wait_sec", 0.0)
    q_avg = queue_metrics.get("avg_completed_wait_sec", 0.0)

    # --- Footer Bar ---
    footer_h = 32
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - footer_h), (w, h), (14, 16, 22), -1)
    cv2.addWeighted(overlay_bot, 0.88, frame, 0.12, 0, frame)

    status_str = f"QUEUE: {q_len} waiting (Max: {int(q_wait)}s, Avg Wait: {int(q_avg)}s)"
    cv2.putText(frame, status_str, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 180, 40), 1, cv2.LINE_AA)

    ctrls = "[SPACE]: Pause  |  [r]: Restart  |  [+/-]: Sampling  |  [q]: Quit"
    cv2.putText(
        frame,
        ctrls,
        (w - 430 if w >= 850 else 280, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )


def main():
    load_dotenv()

    default_vid = os.getenv("DEFAULT_VIDEO_SOURCE", "data/sample_videos/restaurant_cctv_sample.mp4")
    if not os.path.exists(default_vid):
        default_vid = "data/sample_videos/pedestrians_surveillance.mp4"

    parser = argparse.ArgumentParser(
        description="DineSight AI - Phase 3: Spatial ROI, Table-States & Queue Monitoring"
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
        help="Path to ROI JSON configuration file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=2.0,
        help="Table dwell time threshold in seconds",
    )
    parser.add_argument(
        "--release",
        type=float,
        default=2.5,
        help="Table release grace period in seconds",
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
        "--no-loop",
        action="store_true",
        help="Disable automatic looping when video ends",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying a GUI window",
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
    print("      DINESIGHT AI - PHASE 3: TABLE STATES & QUEUE MONITORING")
    print("=" * 70)
    print(f"Video Source : {source_val}")
    print(f"ROI Config   : {args.rois}")
    print(f"Dwell Time   : {args.dwell}s  |  Release Grace: {args.release}s")

    # 1. Initialize Video Reader
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

    # 2. Initialize ROI Manager
    try:
        roi_manager = ROIManager(config_path=args.rois, point_mode="bottom_center")
    except Exception as e:
        print(f"\n[Error] Failed to load ROI configuration: {e}", file=sys.stderr)
        reader.release()
        sys.exit(1)

    # 3. Initialize ByteTracker (YOLOv8 + ByteTrack for person class 0)
    try:
        tracker = ByteTracker(
            model_path=args.model,
            conf_threshold=args.conf,
            target_classes=[0],  # Person class
            device=args.device,
        )
    except Exception as e:
        print(f"\n[Error] Tracker initialization failed: {e}", file=sys.stderr)
        reader.release()
        sys.exit(1)

    # 4. Initialize Table State Engine
    table_ids = list(roi_manager.tables.keys())
    state_engine = TableStateEngine(
        table_ids=table_ids,
        dwell_time_sec=args.dwell,
        release_time_sec=args.release,
    )

    # 5. Initialize Queue Monitor
    queue_monitor = QueueMonitor(min_dwell_sec=args.dwell, loss_grace_sec=args.release)

    meta = reader.get_metadata()
    print(f"Resolution   : {meta['target_resolution']}")
    print(f"Device       : {tracker.device_name}")
    print(f"Tables Configured: {len(table_ids)} {table_ids}\n")

    window_name = "DineSight AI - Phase 3: Spatial Table States & Queue"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    delay_ms = max(1, int(1000.0 / (reader.source_fps or 30.0)))
    paused = False
    current_frame: Optional[np.ndarray] = None
    current_idx = 0

    try:
        while True:
            now = time.time()

            if not paused:
                success, frame, current_idx = reader.read_frame()
                if not success or frame is None:
                    print("\n[DineSight AI] Video playback finished.")
                    break

                # 1. Run ByteTrack multi-object tracking for people
                tracks = tracker.update(frame)

                # 2. Spatial ROI Association (Associate people with tables, queue, aisles)
                h, w = frame.shape[:2]
                spatial_map = roi_manager.associate_tracks(tracks, frame_width=w, frame_height=h)

                # 3. Update Table-State Finite State Machine
                table_occupants = spatial_map["tables"]
                table_states = state_engine.update(table_occupants, now=now)

                # 4. Update Queue Monitor
                queue_tracks = spatial_map["queue"]
                queue_metrics = queue_monitor.update(queue_tracks, now=now)

                current_frame = frame
            else:
                time.sleep(0.03)

            if current_frame is not None:
                annotated = current_frame.copy()

                # A. Draw ROI boundaries, state fills, and table badges
                table_counts = {t_id: len(spatial_map["tables"].get(t_id, [])) for t_id in table_ids}
                annotated = roi_manager.draw_rois(
                    annotated,
                    table_states=table_states,
                    table_occupant_counts=table_counts,
                    queue_count=queue_metrics.get("current_queue_length", 0),
                    alpha=0.25,
                )

                # B. Draw tracked person bounding boxes and IDs
                annotated = tracker.draw_tracks(annotated, tracks, show_center=True)

                # C. Draw Telemetry HUD
                draw_phase3_hud(
                    frame=annotated,
                    frame_idx=current_idx,
                    total_frames=reader.total_frames,
                    pipeline_fps=reader.current_fps,
                    inf_time_ms=tracker.last_track_time_ms,
                    device_name=tracker.device_name,
                    table_states=table_states,
                    queue_metrics=queue_metrics,
                    total_tracked_people=len(tracks),
                    frame_skip=reader.frame_skip,
                    paused=paused,
                )

                if not args.headless:
                    cv2.imshow(window_name, annotated)

            if args.max_frames > 0 and current_idx >= args.max_frames:
                print(f"\n[DineSight AI] Reached target max frames ({args.max_frames}). Completed.")
                break

            if not args.headless:
                key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
                if key in (ord('q'), 27):
                    print("\n[DineSight AI] Exited by user.")
                    break
                elif key in (ord('p'), ord(' ')):
                    paused = not paused
                    print(f"[Playback] {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('r'):
                    reader.restart()
                    tracker.reset()
                    state_engine.reset()
                    queue_monitor.reset()
                    print("[Playback] Video rewound and state machines reset.")
                elif key in (ord('+'), ord('=')):
                    reader.frame_skip += 1
                    print(f"[Config] Frame skip: {reader.frame_skip}")
                elif key in (ord('-'), ord('_')):
                    reader.frame_skip = max(1, reader.frame_skip - 1)
                    print(f"[Config] Frame skip: {reader.frame_skip}")

    finally:
        reader.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

