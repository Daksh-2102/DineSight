"""
DineSight AI - Video Ingestion Module (Module 1)
================================================
Responsible for ingesting video streams from recorded MP4 files,
live webcams, or RTSP network streams using OpenCV.

Key Features:
- Multi-source support (file path, webcam index, or RTSP URL).
- Frame resizing with aspect-ratio preservation or fixed dimensions.
- Configurable frame sampling (skip N frames to conserve computational resources).
- Real-time smoothed FPS (Frames Per Second) calculation.
- Error handling for missing files, corrupt frames, and stream disconnections.
- Seamless video looping option for continuous demonstration.
- Interactive OpenCV preview with on-screen HUD telemetry.
"""

import os
import sys
import time
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class VideoReader:
    """
    OpenCV Video Ingestion and Preprocessing Engine.
    
    Provides a generator-based frame reader with frame resizing,
    sampling, robust error handling, and performance tracking.
    """

    def __init__(
        self,
        source: Union[str, int],
        target_size: Optional[Tuple[int, int]] = (1280, 720),
        frame_skip: int = 1,
        loop: bool = True,
    ):
        """
        Initialize the VideoReader.

        Args:
            source: Path to video file (str), webcam index (int, e.g. 0),
                    or RTSP stream URL (str).
            target_size: Tuple (width, height) to resize frames to. If None,
                         original resolution is preserved.
            frame_skip: Read 1 frame out of every N frames (1 = all frames,
                        2 = every 2nd frame, etc.).
            loop: If True, restart video from beginning when EOF is reached.
                  Ignored for live webcams / live streams.
        """
        self.source = source
        self.target_size = target_size
        self.frame_skip = max(1, int(frame_skip))
        self.loop = loop

        # Determine source type
        self.is_live = False
        if isinstance(source, int):
            self.is_live = True
        elif isinstance(source, str):
            if source.isdigit():
                self.source = int(source)
                self.is_live = True
            elif source.lower().startswith(("rtsp://", "http://", "https://")):
                self.is_live = True

        # Validation for local files
        if not self.is_live and isinstance(self.source, str):
            if not os.path.exists(self.source):
                raise FileNotFoundError(
                    f"[VideoReader Error] Video source file not found: '{self.source}'\n"
                    f"Please verify the file path exists."
                )

        # VideoCapture handle
        self.cap: Optional[cv2.VideoCapture] = None

        # Video metadata
        self.original_width: int = 0
        self.original_height: int = 0
        self.source_fps: float = 0.0
        self.total_frames: int = 0
        self.duration_seconds: float = 0.0

        # Runtime counters
        self.raw_frame_count: int = 0
        self.processed_frame_count: int = 0

        # FPS calculation state (Exponential Moving Average)
        self._last_frame_time: float = 0.0
        self.current_fps: float = 0.0
        self._fps_smoothing_alpha: float = 0.1  # Weight for smoothing

        # Initialize the capture
        self._open_stream()

    def _open_stream(self) -> None:
        """Internal helper to initialize or re-initialize cv2.VideoCapture."""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[VideoReader Error] Unable to open video source: '{self.source}'. "
                f"Verify that the file is not corrupted or the camera is connected."
            )

        # Retrieve video properties
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else 0
        
        if self.source_fps > 0 and self.total_frames > 0:
            self.duration_seconds = self.total_frames / self.source_fps
        else:
            self.duration_seconds = 0.0

        self.raw_frame_count = 0
        self.processed_frame_count = 0
        self._last_frame_time = time.perf_counter()

    def get_metadata(self) -> dict:
        """Return a dictionary containing video stream metadata."""
        return {
            "source": str(self.source),
            "is_live": self.is_live,
            "original_resolution": f"{self.original_width}x{self.original_height}",
            "target_resolution": f"{self.target_size[0]}x{self.target_size[1]}" if self.target_size else "Original",
            "source_fps": round(self.source_fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "frame_skip": self.frame_skip,
            "loop": self.loop,
        }

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target_size if specified."""
        if self.target_size is None:
            return frame

        target_w, target_h = self.target_size
        if frame.shape[1] == target_w and frame.shape[0] == target_h:
            return frame

        # cv2.INTER_LINEAR provides the best speed-to-quality balance for video
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _update_fps(self) -> float:
        """
        Calculate instant and smoothed processing FPS using Exponential Moving Average.
        
        Formula:
            Instant FPS = 1.0 / delta_time
            Smoothed FPS = alpha * Instant_FPS + (1 - alpha) * Previous_Smoothed_FPS
        """
        now = time.perf_counter()
        delta = now - self._last_frame_time
        self._last_frame_time = now

        if delta > 0:
            instant_fps = 1.0 / delta
            if self.current_fps == 0.0:
                self.current_fps = instant_fps
            else:
                self.current_fps = (
                    self._fps_smoothing_alpha * instant_fps
                    + (1.0 - self._fps_smoothing_alpha) * self.current_fps
                )
        return self.current_fps

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Read a single processed frame, adhering to the frame_skip interval.

        Returns:
            Tuple (success, frame, raw_frame_index):
                success (bool): True if a frame was successfully read.
                frame (np.ndarray or None): The resized frame image.
                raw_frame_index (int): The current frame index in the source.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, self.raw_frame_count

        while True:
            success, frame = self.cap.read()
            self.raw_frame_count += 1

            if not success:
                # End of video reached or frame read error
                if not self.is_live and self.loop:
                    # Rewind to frame 0 and retry
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.raw_frame_count = 0
                    success, frame = self.cap.read()
                    if not success:
                        return False, None, self.raw_frame_count
                else:
                    return False, None, self.raw_frame_count

            # Apply frame sampling: only yield if this frame matches the skip interval
            if (self.raw_frame_count - 1) % self.frame_skip == 0:
                self.processed_frame_count += 1
                self._update_fps()
                resized_frame = self._resize_frame(frame)
                return True, resized_frame, self.raw_frame_count

    def frames(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Generator yielding frames continuously.

        Yields:
            Tuple (frame_number, frame, current_fps)
        """
        try:
            while True:
                success, frame, frame_idx = self.read_frame()
                if not success or frame is None:
                    break
                yield frame_idx, frame, self.current_fps
        finally:
            self.release()

    def restart(self) -> None:
        """Restart playback from the first frame."""
        if self.cap is not None and not self.is_live:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.raw_frame_count = 0
            self.processed_frame_count = 0
            self._last_frame_time = time.perf_counter()

    def release(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def run_preview(
        self,
        window_name: str = "DineSight AI - Video Ingestion Preview",
        headless: bool = False,
        max_frames: int = 0,
    ) -> None:
        """
        Run an interactive window displaying the video stream with HUD telemetry.

        Controls:
            [q] or [ESC] : Quit
            [SPACE] or [p]: Pause / Resume
            [r]          : Rewind / Restart video
            [+] / [-]    : Increase / Decrease frame skip rate
        """
        print(f"\n[DineSight AI] Starting Video Ingestion Preview...")
        print(f"Source: {self.source}")
        meta = self.get_metadata()
        for k, v in meta.items():
            print(f"  - {k}: {v}")

        if not headless:
            print("\nKeyboard Controls:")
            print("  [q] or [ESC]  : Quit")
            print("  [SPACE] / [p] : Pause / Resume")
            print("  [r]           : Restart video")
            print("  [+] / [-]     : Increase / Decrease frame skip")

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if self.target_size:
                cv2.resizeWindow(window_name, self.target_size[0], self.target_size[1])

        paused = False
        current_frame: Optional[np.ndarray] = None
        current_idx: int = 0

        # Delay calculated to approximate native video playback speed
        delay_ms = max(1, int(1000.0 / (self.source_fps or 30.0)))

        try:
            while True:
                if not paused:
                    success, frame, current_idx = self.read_frame()
                    if not success or frame is None:
                        print("\n[VideoReader] Stream ended.")
                        break
                    current_frame = frame
                else:
                    # When paused, keep displaying the current frame without advancing
                    time.sleep(0.03)

                if current_frame is not None:
                    display_frame = current_frame.copy()
                    self._draw_hud(display_frame, current_idx, paused)
                    if not headless:
                        cv2.imshow(window_name, display_frame)

                if max_frames > 0 and current_idx >= max_frames:
                    print(f"\n[VideoReader] Reached target max frames ({max_frames}). Completed.")
                    break

                # Handle keyboard input
                if not headless:
                    key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
                    if key in (ord('q'), 27):  # 'q' or ESC
                        print("\n[VideoReader] Exiting preview by user request.")
                        break
                    elif key in (ord('p'), ord(' ')):  # 'p' or SPACE
                        paused = not paused
                        state_str = "PAUSED" if paused else "RESUMED"
                        print(f"[Playback] {state_str}")
                    elif key == ord('r'):  # 'r'
                        self.restart()
                        print("[Playback] Rewound to beginning.")
                    elif key in (ord('+'), ord('=')):  # '+'
                        self.frame_skip += 1
                        print(f"[Config] Frame skip increased to: {self.frame_skip}")
                    elif key in (ord('-'), ord('_')):  # '-'
                        self.frame_skip = max(1, self.frame_skip - 1)
                        print(f"[Config] Frame skip decreased to: {self.frame_skip}")

        finally:
            self.release()
            if not headless:
                cv2.destroyAllWindows()

    def _draw_hud(self, frame: np.ndarray, frame_idx: int, paused: bool) -> None:
        """Draw a professional semi-transparent heads-up display (HUD) on the frame."""
        h, w = frame.shape[:2]

        # Draw semi-transparent header bar
        header_height = 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 24, 30), -1)
        # Blend overlay (alpha = 0.85)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Brand label
        cv2.putText(
            frame,
            "DineSight AI | Video Ingestion",
            (15, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 215, 255),  # DineSight gold/amber
            2,
            cv2.LINE_AA,
        )

        # Live telemetry: FPS, Resolution, Frame Index
        fps_text = f"FPS: {self.current_fps:4.1f}"
        res_text = f"Res: {w}x{h}"
        skip_text = f"Sampling: 1/{self.frame_skip}"
        
        if self.total_frames > 0:
            progress_pct = (frame_idx / self.total_frames) * 100
            frame_text = f"Frame: {frame_idx}/{self.total_frames} ({progress_pct:4.1f}%)"
        else:
            frame_text = f"Frame: {frame_idx} (Live)"

        telemetry = f"{fps_text}  |  {res_text}  |  {skip_text}  |  {frame_text}"
        cv2.putText(
            frame,
            telemetry,
            (w - 620 if w >= 1000 else 15, 33 if w >= 1000 else 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        # Bottom status bar for controls & playback state
        footer_height = 30
        overlay_foot = frame.copy()
        cv2.rectangle(overlay_foot, (0, h - footer_height), (w, h), (15, 17, 23), -1)
        cv2.addWeighted(overlay_foot, 0.85, frame, 0.15, 0, frame)

        status_text = "PAUSED [Press SPACE to resume]" if paused else "PLAYING [Press SPACE to pause]"
        status_color = (0, 165, 255) if paused else (0, 255, 127)  # Orange or Green
        cv2.putText(
            frame,
            status_text,
            (15, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
            cv2.LINE_AA,
        )

        controls_tip = "[q]: Quit  |  [r]: Rewind  |  [+/-]: Adjust Sampling"
        cv2.putText(
            frame,
            controls_tip,
            (w - 380, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="DineSight AI - Video Ingestion Preview")
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("DEFAULT_VIDEO_SOURCE", "data/sample_videos/restaurant_sample.mp4"),
        help="Path to video file, webcam index (e.g. 0), or RTSP URL",
    )
    parser.add_argument("--width", type=int, default=int(os.getenv("FRAME_WIDTH", 1280)), help="Target width")
    parser.add_argument("--height", type=int, default=int(os.getenv("FRAME_HEIGHT", 720)), help="Target height")
    parser.add_argument(
        "--skip",
        type=int,
        default=int(os.getenv("FRAME_SKIP_INTERVAL", 1)),
        help="Frame skip interval (1 = all frames, 2 = every 2nd frame)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable automatic looping when video reaches end",
    )

    args = parser.parse_args()

    # Convert numeric string to int for webcam indices
    source_val: Union[str, int] = int(args.source) if args.source.isdigit() else args.source

    try:
        reader = VideoReader(
            source=source_val,
            target_size=(args.width, args.height),
            frame_skip=args.skip,
            loop=not args.no_loop,
        )
        reader.run_preview()
    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)
