"""
DineSight AI - Phase 2 Runner Script
====================================
Interactive launcher for YOLOv8 Object Detection and ByteTrack Multi-Object Tracking.

Features:
- Live video ingestion using Phase 1 VideoReader.
- Pretrained YOLOv8 Nano inference for person detection.
- ByteTrack persistent multi-object tracking.
- Real-time telemetry: Device (CPU vs GPU), inference latency, pipeline FPS, active people count.
- Interactive keyboard shortcuts (pause, rewind, frame sampling, quit).

Usage:
    python run_phase2.py
    python run_phase2.py --source 0                         # Live webcam
    python run_phase2.py --source path/to/video.mp4         # Custom video
    python run_phase2.py --conf 0.40 --skip 1               # Custom confidence & sampling
    python run_phase2.py --device cpu                       # Force CPU mode
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
from backend.app.core.tracker import ByteTracker


def draw_phase2_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    pipeline_fps: float,
    inf_time_ms: float,
    device_name: str,
    active_people: int,
    total_unique_seen: int,
    frame_skip: int,
    paused: bool,
) -> None:
    """Render a comprehensive telemetry HUD for Phase 2."""
    h, w = frame.shape[:2]

    # Top Header Bar
    header_h = 60
    overlay_top = frame.copy()
    cv2.rectangle(overlay_top, (0, 0), (w, header_h), (18, 22, 28), -1)
    cv2.addWeighted(overlay_top, 0.88, frame, 0.12, 0, frame)

    # Title
    cv2.putText(
        frame,
        "DineSight AI | YOLOv8 + ByteTrack",
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),  # DineSight Gold
        2,
        cv2.LINE_AA,
    )

    # Top Subtitle / Device info
    device_color = (0, 255, 127) if "GPU" in device_name else (200, 200, 200)
    dev_str = f"Device: {device_name}"
    cv2.putText(
        frame,
        dev_str,
        (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        device_color,
        1,
        cv2.LINE_AA,
    )

    # Right-aligned metrics
    inf_fps = 1000.0 / inf_time_ms if inf_time_ms > 0 else 0.0
    line1 = f"Inference: {inf_time_ms:4.1f}ms ({inf_fps:3.0f} FPS)  |  Pipeline: {pipeline_fps:4.1f} FPS"
    progress = f"Frame: {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame: {frame_idx}"
    line2 = f"Active People: {active_people}  |  Total Seen: {total_unique_seen}  |  Sampling: 1/{frame_skip}  |  {progress}"

    cv2.putText(
        frame,
        line1,
        (w - 600 if w >= 950 else 15, 26 if w >= 950 else 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        line2,
        (w - 600 if w >= 950 else 15, 48 if w >= 950 else 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 210, 230),
        1,
        cv2.LINE_AA,
    )

    # Bottom Controls Bar
    footer_h = 30
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - footer_h), (w, h), (14, 16, 22), -1)
    cv2.addWeighted(overlay_bot, 0.88, frame, 0.12, 0, frame)

    status_str = "PAUSED [Press SPACE to resume]" if paused else "LIVE TRACKING [Press SPACE to pause]"
    status_col = (0, 165, 255) if paused else (0, 255, 127)
    cv2.putText(
        frame,
        status_str,
        (15, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        status_col,
        1,
        cv2.LINE_AA,
    )

    tips = "[SPACE]: Pause  |  [r]: Restart  |  [+/-]: Sampling  |  [q]: Quit"
    cv2.putText(
        frame,
        tips,
        (w - 430 if w >= 800 else 240, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )


def main():
    load_dotenv()

    # Determine default video: prefer pedestrians_surveillance.mp4 if present
    default_vid = os.getenv("DEFAULT_VIDEO_SOURCE", "data/sample_videos/restaurant_cctv_sample.mp4")
    if not os.path.exists(default_vid):
        default_vid = "data/sample_videos/pedestrians_surveillance.mp4"

    parser = argparse.ArgumentParser(
        description="DineSight AI - Phase 2: YOLOv8 Detection & ByteTrack Tracking"
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
        help="Path to YOLOv8 weights (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for person detection (default 0.35)",
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
        help="Frame skip interval (1 = process all frames, 2 = every 2nd frame)",
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

    print("\n" + "=" * 65)
    print("      DINESIGHT AI - PHASE 2: DETECTION & TRACKING")
    print("=" * 65)
    print(f"Video Source : {source_val}")
    print(f"Model Weights: {args.model}")
    print(f"Confidence   : {args.conf}")

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

    # 2. Initialize ByteTracker (with YOLOv8n)
    try:
        tracker = ByteTracker(
            model_path=args.model,
            conf_threshold=args.conf,
            target_classes=[0],  # Person class only
            device=args.device,
        )
    except Exception as e:
        print(f"\n[Error] Failed to initialize tracker: {e}", file=sys.stderr)
        reader.release()
        sys.exit(1)

    meta = reader.get_metadata()
    print(f"Resolution   : {meta['target_resolution']}")
    print(f"Device       : {tracker.device_name}")
    print(f"Tracker      : ByteTrack (Ultralytics)\n")
    print("Controls:")
    print("  [SPACE] / [p] : Pause / Resume")
    print("  [r]           : Rewind to beginning")
    print("  [+] / [-]     : Increase / Decrease frame skip")
    print("  [q] or [ESC]  : Quit\n")

    window_name = "DineSight AI - Phase 2: YOLOv8 + ByteTrack"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    delay_ms = max(1, int(1000.0 / (reader.source_fps or 30.0)))
    paused = False
    current_frame: Optional[np.ndarray] = None
    current_tracks = []
    current_idx = 0

    try:
        while True:
            if not paused:
                success, frame, current_idx = reader.read_frame()
                if not success or frame is None:
                    print("\n[DineSight AI] Video playback completed.")
                    break

                # Run ByteTrack multi-object tracking on the frame
                current_tracks = tracker.update(frame)
                current_frame = frame
            else:
                time.sleep(0.03)

            if current_frame is not None:
                # Annotate tracked bounding boxes and IDs
                annotated = tracker.draw_tracks(current_frame, current_tracks)

                # Draw Phase 2 telemetry HUD
                draw_phase2_hud(
                    frame=annotated,
                    frame_idx=current_idx,
                    total_frames=reader.total_frames,
                    pipeline_fps=reader.current_fps,
                    inf_time_ms=tracker.last_track_time_ms,
                    device_name=tracker.device_name,
                    active_people=len(current_tracks),
                    total_unique_seen=tracker.total_unique_tracks_seen,
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
                    print("\n[DineSight AI] Exit requested by user.")
                    break
                elif key in (ord('p'), ord(' ')):
                    paused = not paused
                    print(f"[Playback] {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('r'):
                    reader.restart()
                    tracker.reset()
                    print("[Playback] Rewound and tracker state reset.")
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

