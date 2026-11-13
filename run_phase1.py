"""
DineSight AI - Phase 1 Runner Script
====================================
Quick launcher for the Video Ingestion & Telemetry module.

Usage:
    python run_phase1.py
    python run_phase1.py --source 0               # Use webcam
    python run_phase1.py --source path/to/cctv.mp4 # Custom video
    python run_phase1.py --skip 2                 # Sample every 2nd frame
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.core.video_reader import VideoReader


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="DineSight AI - Phase 1: Video Ingestion & Telemetry"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("DEFAULT_VIDEO_SOURCE", "data/sample_videos/restaurant_sample.mp4"),
        help="Path to video file, webcam index (e.g. 0), or RTSP URL",
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
        "--no-loop",
        action="store_true",
        help="Disable automatic looping when video ends",
    )

    args = parser.parse_args()

    # If the user passed a numeric string, treat it as an integer camera index
    source_val = int(args.source) if args.source.isdigit() else args.source

    print("\n" + "=" * 60)
    print("        DINESIGHT AI - VIDEO INGESTION ENGINE")
    print("=" * 60)

    try:
        reader = VideoReader(
            source=source_val,
            target_size=(args.width, args.height),
            frame_skip=args.skip,
            loop=not args.no_loop,
        )
        reader.run_preview(window_name="DineSight AI - Video Ingestion (Phase 1)")
    except FileNotFoundError as fnf_err:
        print(f"\n[Error] {fnf_err}")
        print("Tip: You can regenerate the synthetic sample video with:")
        print("  python -m backend.app.core.generate_sample_video")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Unexpected Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
