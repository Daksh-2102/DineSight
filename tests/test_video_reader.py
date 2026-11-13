"""
DineSight AI - Video Ingestion Automated Test Suite
===================================================
Tests all features of the VideoReader module:
1. Metadata reading
2. Frame dimensions and resizing
3. Frame sampling / skipping logic
4. FPS calculation
5. Loop vs non-loop EOF behavior
6. Error handling for missing and invalid files
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.video_reader import VideoReader


class TestVideoReader(unittest.TestCase):
    SAMPLE_VIDEO = "data/sample_videos/restaurant_sample.mp4"

    def setUp(self):
        self.assertTrue(
            os.path.exists(self.SAMPLE_VIDEO),
            f"Sample video does not exist at {self.SAMPLE_VIDEO}. Run generate_sample_video.py first.",
        )

    def test_01_metadata(self):
        """Test video metadata extraction."""
        with VideoReader(self.SAMPLE_VIDEO, target_size=(1280, 720)) as reader:
            meta = reader.get_metadata()
            self.assertEqual(meta["original_resolution"], "1280x720")
            self.assertEqual(meta["target_resolution"], "1280x720")
            self.assertEqual(meta["total_frames"], 450)
            self.assertAlmostEqual(meta["source_fps"], 30.0, delta=1.0)
            self.assertAlmostEqual(meta["duration_seconds"], 15.0, delta=1.0)
            print(" [PASS] Metadata extraction test")

    def test_02_frame_resizing(self):
        """Test frame resizing to a different target resolution."""
        target_w, target_h = 640, 360
        with VideoReader(self.SAMPLE_VIDEO, target_size=(target_w, target_h)) as reader:
            success, frame, idx = reader.read_frame()
            self.assertTrue(success)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (target_h, target_w, 3))
            print(" [PASS] Frame resizing test (640x360)")

    def test_03_frame_sampling(self):
        """Test frame skip logic (e.g. skip=3 yields 1 out of every 3 frames)."""
        skip_val = 3
        with VideoReader(self.SAMPLE_VIDEO, frame_skip=skip_val, loop=False) as reader:
            yielded = 0
            for idx, frame, fps in reader.frames():
                yielded += 1
                if yielded >= 30:  # test first 30 sampled frames
                    break
            # Out of first ~90 raw frames, exactly 30 should be sampled
            self.assertEqual(yielded, 30)
            self.assertGreater(reader.raw_frame_count, 85)
            print(f" [PASS] Frame sampling test (raw={reader.raw_frame_count}, sampled={yielded})")

    def test_04_fps_tracking(self):
        """Test that FPS calculation is updated and positive."""
        with VideoReader(self.SAMPLE_VIDEO, frame_skip=1) as reader:
            for _ in range(10):
                success, frame, idx = reader.read_frame()
                self.assertTrue(success)
            self.assertGreater(reader.current_fps, 0.0)
            print(f" [PASS] FPS tracking test (current_fps={reader.current_fps:.1f})")

    def test_05_eof_termination(self):
        """Test that loop=False cleanly terminates at end of video."""
        with VideoReader(self.SAMPLE_VIDEO, frame_skip=10, loop=False) as reader:
            count = 0
            for idx, frame, fps in reader.frames():
                count += 1
            # 450 frames / 10 skip = 45 frames
            self.assertEqual(count, 45)
            # Next read should return False
            success, frame, _ = reader.read_frame()
            self.assertFalse(success)
            self.assertIsNone(frame)
            print(" [PASS] Clean EOF termination test")

    def test_06_error_handling_missing_file(self):
        """Test that missing files raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            VideoReader("data/sample_videos/non_existent_file.mp4")
        print(" [PASS] Missing file error handling test")


if __name__ == "__main__":
    print("\nRunning DineSight AI - Phase 1 Test Suite...\n" + "=" * 50)
    unittest.main(verbosity=2)

