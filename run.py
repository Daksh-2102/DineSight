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
    frame_skip: int = 1,
    paused: bool = False,
) -> None:
    """Render a unified HUD header and footer across all operational modes."""
    h, w = frame.shape[:2]

    # --- Header Bar ---
    header_h = 65
    overlay_top = frame.copy()
    cv2.rectangle(overlay_top, (0, 0), (w, header_h), (16, 20, 26), -1)
    cv2.addWeighted(overlay_top, 0.88, frame, 0.12, 0, frame)

    mode_label = {
        "raw": "RAW VIDEO INGESTION (Phase 1)",
        "detect": "YOLOv8 DETECTION ONLY (Phase 2)",
        "track": "YOLOv8 + BYTETRACK TRACKING (Phase 2)",
        "full": "FULL RESTAURANT ANALYTICS (Phase 1+2+3)",
    }.get(mode, mode.upper())

    # Title & Mode
    cv2.putText(
        frame,
        f"DineSight AI | {mode_label}",
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    # Device & Settings
    dev_str = f"Device: {device_name}  |  Sampling: 1/{frame_skip}" if mode != "raw" else f"Mode: Passthrough  |  Sampling: 1/{frame_skip}"
    dev_col = (0, 255, 127) if "GPU" in device_name else (200, 200, 200)
    cv2.putText(frame, dev_str, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, dev_col, 1, cv2.LINE_AA)

    # Performance & Object Telemetry
    inf_fps = (1000.0 / inf_time_ms) if inf_time_ms > 0 else 0.0
    if mode == "full" and table_states is not None:
        total_tables = len(table_states)
        occ = sum(1 for s in table_states.values() if s != "VACANT")
        pct = int((occ / total_tables) * 100) if total_tables > 0 else 0
        perf_line = f"Inf: {inf_time_ms:4.1f}ms ({inf_fps:3.0f} FPS)  |  Pipeline: {pipeline_fps:4.1f} FPS  |  People: {count}"
        count_line = f"Occupancy: {occ}/{total_tables} ({pct}%)  |  Queue: {queue_metrics.get('current_queue_length', 0)} waiting"
    elif mode == "track":
        perf_line = f"Inf: {inf_time_ms:4.1f}ms ({inf_fps:3.0f} FPS)  |  Pipeline: {pipeline_fps:4.1f} FPS"
        count_line = f"Active: {count}  |  Total Seen: {total_unique}  |  Frame: {frame_idx}/{total_frames}"
    elif mode == "detect":
        perf_line = f"Inf: {inf_time_ms:4.1f}ms ({inf_fps:3.0f} FPS)  |  Pipeline: {pipeline_fps:4.1f} FPS"
        count_line = f"People Detected: {count}  |  Frame: {frame_idx}/{total_frames}"
    else:  # raw
        perf_line = f"Pipeline: {pipeline_fps:4.1f} FPS"
        count_line = f"Frame: {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame: {frame_idx}"

    cv2.putText(
        frame,
        perf_line,
        (w - 650 if w >= 1000 else 15, 26 if w >= 1000 else 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        count_line,
        (w - 650 if w >= 1000 else 15, 50 if w >= 1000 else 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 215, 240),
        1,
        cv2.LINE_AA,
    )

    # --- Footer Bar ---
    footer_h = 30
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - footer_h), (w, h), (14, 16, 22), -1)
    cv2.addWeighted(overlay_bot, 0.88, frame, 0.12, 0, frame)

    status_str = "PAUSED [SPACE to resume]" if paused else "ACTIVE [SPACE to pause]"
    status_col = (0, 165, 255) if paused else (0, 255, 127)
    cv2.putText(frame, status_str, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_col, 1, cv2.LINE_AA)

    tips = "[SPACE]: Pause  |  [r]: Restart  |  [+/-]: Sampling  |  [m]: Toggle Mode  |  [q]: Quit"
    cv2.putText(
        frame,
        tips,
        (w - 530 if w >= 950 else 220, h - 9),
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
        default=0.35,
        help="Confidence threshold for person detection",
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
    device_name = "CPU"

    # Always initialize ROI and state engines for full mode
    if os.path.exists(args.rois):
        try:
            roi_manager = ROIManager(config_path=args.rois, point_mode="bottom_center")
            table_ids = list(roi_manager.tables.keys())
            state_engine = TableStateEngine(table_ids=table_ids)
            queue_monitor = QueueMonitor()
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

                # Mode branching
                if current_mode == "detect":
                    if detector is None:
                        detector = YOLODetector(model_path=args.model, conf_threshold=args.conf, target_classes=[0], device=args.device)
                    dets = detector.detect(current_frame)
                    annotated = detector.draw_detections(annotated, dets)
                    count = len(dets)
                    inf_time_ms = detector.last_inference_time_ms
                    device_name = detector.device_name

                elif current_mode in ("track", "full"):
                    if tracker is None:
                        tracker = ByteTracker(model_path=args.model, conf_threshold=args.conf, target_classes=[0], device=args.device)
                    trks = tracker.update(current_frame)
                    count = len(trks)
                    total_unique = tracker.total_unique_tracks_seen
                    inf_time_ms = tracker.last_track_time_ms
                    device_name = tracker.device_name

                    if current_mode == "full" and roi_manager and state_engine and queue_monitor:
                        h, w = current_frame.shape[:2]
                        spatial_map = roi_manager.associate_tracks(trks, frame_width=w, frame_height=h)
                        table_states = state_engine.update(spatial_map["tables"], now=now)
                        queue_metrics = queue_monitor.update(spatial_map["queue"], now=now)

                        # Draw ROIs
                        table_counts = {t_id: len(spatial_map["tables"].get(t_id, [])) for t_id in roi_manager.tables}
                        annotated = roi_manager.draw_rois(
                            annotated,
                            table_states=table_states,
                            table_occupant_counts=table_counts,
                            queue_count=queue_metrics.get("current_queue_length", 0),
                        )

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
                    frame_skip=reader.frame_skip,
                    paused=paused,
                )

                cv2.imshow(window_name, annotated)

            # Keyboard controls
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
