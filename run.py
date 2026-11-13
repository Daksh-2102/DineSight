"""
DineSight AI - Master Application Runner
========================================
Unifies Phase 1 (Video Ingestion) and Phase 2 (YOLOv8 Detection & ByteTrack Tracking)
into a single, configurable interface.

Modes:
    1. 'track'  (Default): Full pipeline (Video Ingestion + YOLOv8 + ByteTrack IDs).
    2. 'detect': Detection only (Video Ingestion + YOLOv8 Bounding Boxes without tracking).
    3. 'raw'   : Raw video ingestion (Phase 1 baseline without AI models).

Usage:
    python run.py                          # Full AI tracking pipeline
    python run.py --mode raw               # Test video feed only (Phase 1)
    python run.py --mode detect            # YOLOv8 detection only
    python run.py --source 0               # Run on live webcam
    python run.py --skip 2                 # Sample every 2nd frame
"""

import os
import sys
import time
import argparse
from typing import Optional, Union
import cv2
import numpy as np
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.core.video_reader import VideoReader
from backend.app.core.detector import YOLODetector
from backend.app.core.tracker import ByteTracker


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
    frame_skip: int,
    paused: bool,
) -> None:
    """Render a unified HUD header and footer across all modes."""
    h, w = frame.shape[:2]

    # --- Header Bar ---
    header_h = 60
    overlay_top = frame.copy()
    cv2.rectangle(overlay_top, (0, 0), (w, header_h), (18, 22, 28), -1)
    cv2.addWeighted(overlay_top, 0.88, frame, 0.12, 0, frame)

    mode_label = {
        "raw": "RAW VIDEO INGESTION",
        "detect": "YOLOv8 DETECTION ONLY",
        "track": "YOLOv8 + BYTETRACK TRACKING",
    }.get(mode, mode.upper())

    # Title & Mode
    cv2.putText(
        frame,
        f"DineSight AI | {mode_label}",
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    # Device & Settings
    dev_str = f"Device: {device_name}" if mode != "raw" else "Mode: Direct Passthrough"
    dev_col = (0, 255, 127) if "GPU" in device_name else (200, 200, 200)
    cv2.putText(frame, dev_str, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dev_col, 1, cv2.LINE_AA)

    # Performance & Object Telemetry
    inf_fps = (1000.0 / inf_time_ms) if inf_time_ms > 0 else 0.0
    if mode == "raw":
        perf_line = f"Pipeline: {pipeline_fps:4.1f} FPS  |  Sampling: 1/{frame_skip}"
        count_line = f"Frame: {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame: {frame_idx}"
    elif mode == "detect":
        perf_line = f"Inference: {inf_time_ms:4.1f}ms ({inf_fps:3.0f} FPS)  |  Pipeline: {pipeline_fps:4.1f} FPS"
        count_line = f"People Detected: {count}  |  Frame: {frame_idx}/{total_frames}"
    else:  # track
        perf_line = f"Inference: {inf_time_ms:4.1f}ms ({inf_fps:3.0f} FPS)  |  Pipeline: {pipeline_fps:4.1f} FPS"
        count_line = f"Active: {count}  |  Total Seen: {total_unique}  |  Frame: {frame_idx}/{total_frames}"

    cv2.putText(
        frame,
        perf_line,
        (w - 600 if w >= 950 else 15, 26 if w >= 950 else 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        count_line,
        (w - 600 if w >= 950 else 15, 48 if w >= 950 else 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 210, 230),
        1,
        cv2.LINE_AA,
    )

    # --- Footer Bar ---
    footer_h = 30
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - footer_h), (w, h), (14, 16, 22), -1)
    cv2.addWeighted(overlay_bot, 0.88, frame, 0.12, 0, frame)

    status_str = "PAUSED [SPACE to resume]" if paused else "RUNNING [SPACE to pause]"
    status_col = (0, 165, 255) if paused else (0, 255, 127)
    cv2.putText(frame, status_str, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_col, 1, cv2.LINE_AA)

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

    # Default video selection
    default_vid = "data/sample_videos/pedestrians_surveillance.mp4"
    if not os.path.exists(default_vid):
        default_vid = os.getenv("DEFAULT_VIDEO_SOURCE", "data/sample_videos/restaurant_sample.mp4")

    parser = argparse.ArgumentParser(
        description="DineSight AI - Master Application Runner"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="track",
        choices=["raw", "detect", "track"],
        help="Operation mode: 'raw' (Phase 1), 'detect' (YOLO only), 'track' (YOLO + ByteTrack)",
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
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for person detection (default: 0.35)",
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

    print("\n" + "=" * 65)
    print("             DINESIGHT AI - MASTER PIPELINE")
    print("=" * 65)
    print(f"Active Mode  : {args.mode.upper()}")
    print(f"Video Source : {source_val}")

    # 1. Initialize Phase 1 Video Ingestion
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

    # 2. Initialize AI Engines based on mode
    detector = None
    tracker = None
    device_name = "CPU"

    if args.mode in ("detect", "track"):
        print(f"Model Weights: {args.model}")
        print(f"Confidence   : {args.conf}")

        if args.mode == "detect":
            detector = YOLODetector(
                model_path=args.model,
                conf_threshold=args.conf,
                target_classes=[0],
                device=args.device,
            )
            device_name = detector.device_name
        else:  # track
            tracker = ByteTracker(
                model_path=args.model,
                conf_threshold=args.conf,
                target_classes=[0],
                device=args.device,
            )
            device_name = tracker.device_name

    meta = reader.get_metadata()
    print(f"Resolution   : {meta['target_resolution']}")
    print(f"Compute Dev  : {device_name}\n")
    print("Keyboard Shortcuts:")
    print("  [SPACE] / [p] : Pause / Resume")
    print("  [r]           : Restart video and reset tracking")
    print("  [+] / [-]     : Dynamically adjust frame sampling")
    print("  [m]           : Toggle modes (raw -> detect -> track)")
    print("  [q] or [ESC]  : Quit\n")

    window_name = "DineSight AI - Integrated System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    delay_ms = max(1, int(1000.0 / (reader.source_fps or 30.0)))
    paused = False
    current_frame: Optional[np.ndarray] = None
    current_mode = args.mode
    current_idx = 0

    try:
        while True:
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

                # Branch by mode
                if current_mode == "detect":
                    if detector is None:
                        detector = YOLODetector(
                            model_path=args.model,
                            conf_threshold=args.conf,
                            target_classes=[0],
                            device=args.device,
                        )
                    dets = detector.detect(current_frame)
                    annotated = detector.draw_detections(annotated, dets)
                    count = len(dets)
                    inf_time_ms = detector.last_inference_time_ms
                    device_name = detector.device_name

                elif current_mode == "track":
                    if tracker is None:
                        tracker = ByteTracker(
                            model_path=args.model,
                            conf_threshold=args.conf,
                            target_classes=[0],
                            device=args.device,
                        )
                    trks = tracker.update(current_frame)
                    annotated = tracker.draw_tracks(annotated, trks)
                    count = len(trks)
                    total_unique = tracker.total_unique_tracks_seen
                    inf_time_ms = tracker.last_track_time_ms
                    device_name = tracker.device_name

                # Draw Unified Master HUD
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
                print("[Playback] Rewound and tracker state reset.")
            elif key in (ord('+'), ord('=')):
                reader.frame_skip += 1
                print(f"[Config] Frame skip: {reader.frame_skip}")
            elif key in (ord('-'), ord('_')):
                reader.frame_skip = max(1, reader.frame_skip - 1)
                print(f"[Config] Frame skip: {reader.frame_skip}")
            elif key == ord('m'):
                # Cycle modes dynamically: raw -> detect -> track -> raw
                modes = ["raw", "detect", "track"]
                current_mode = modes[(modes.index(current_mode) + 1) % len(modes)]
                print(f"[Mode Switch] Active mode: {current_mode.upper()}")

    finally:
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

