"""
DineSight AI - ByteTrack Tracker Test Suite (Phase 2)
====================================================
Tests for ByteTracker:
1. Tracker initialization and device resolution.
2. Handling of empty frames.
3. Tracking ID assignment on real video frames (Integration test).
4. Temporal ID persistence across consecutive frames (Integration test).
5. Tracker state reset behavior.
6. Track visualization drawing utility.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.tracker import ByteTracker
from backend.app.core.video_reader import VideoReader


class TestByteTracker(unittest.TestCase):
    MODEL_PATH = "data/models/yolov8n.pt"
    SAMPLE_VIDEO = "data/sample_videos/pedestrians_surveillance.mp4"

    def setUp(self):
        self.assertTrue(
            os.path.exists(self.MODEL_PATH),
            f"Model weights not found at {self.MODEL_PATH}.",
        )
        self.assertTrue(
            os.path.exists(self.SAMPLE_VIDEO),
            f"Sample video not found at {self.SAMPLE_VIDEO}.",
        )

    def test_01_tracker_initialization(self):
        """Test ByteTracker initialization."""
        tracker_default = ByteTracker(model_path=self.MODEL_PATH)
        self.assertIsNotNone(tracker_default.model)
        self.assertTrue(
            tracker_default.tracker_type in ("config/bytetrack_dinesight.yaml", "bytetrack.yaml")
        )

        tracker_custom = ByteTracker(model_path=self.MODEL_PATH, tracker_type="bytetrack.yaml")
        self.assertEqual(tracker_custom.tracker_type, "bytetrack.yaml")
        print(" [PASS] Test 1: ByteTracker initialization verified (default & explicit config)")

    def test_02_empty_frame_handling(self):
        """Test that a blank frame returns an empty track list without crashing."""
        tracker = ByteTracker(model_path=self.MODEL_PATH, conf_threshold=0.5)
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = tracker.update(black_frame)
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 0)
        print(" [PASS] Test 2: Blank frame returns empty track list gracefully")

    def test_03_track_id_assignment(self):
        """[Integration Test] Verify that tracked people are assigned positive integer track IDs."""
        with VideoReader(self.SAMPLE_VIDEO) as reader:
            success, frame, _ = reader.read_frame()
            self.assertTrue(success)

            tracker = ByteTracker(model_path=self.MODEL_PATH, conf_threshold=0.35)
            tracks = tracker.update(frame)

            self.assertIsInstance(tracks, list)
            self.assertGreater(len(tracks), 0, "Expected at least 1 tracked person")

            for trk in tracks:
                self.assertIn("track_id", trk)
                self.assertIsInstance(trk["track_id"], int)
                self.assertGreaterEqual(trk["track_id"], 1)
                self.assertIn("bbox", trk)
                self.assertIn("confidence", trk)
                self.assertIn("center", trk)

            print(f" [PASS] Test 3: Tracking ID assignment verified on {len(tracks)} people in frame 1")

    def test_04_track_persistence(self):
        """[Integration Test] Verify that a person maintains the same track ID across consecutive frames."""
        with VideoReader(self.SAMPLE_VIDEO) as reader:
            tracker = ByteTracker(model_path=self.MODEL_PATH, conf_threshold=0.35)

            ids_frame_1 = set()
            ids_frame_2 = set()

            # Process Frame 1
            success, frame1, _ = reader.read_frame()
            self.assertTrue(success)
            tracks1 = tracker.update(frame1)
            for t in tracks1:
                ids_frame_1.add(t["track_id"])

            # Process Frame 2
            success, frame2, _ = reader.read_frame()
            self.assertTrue(success)
            tracks2 = tracker.update(frame2)
            for t in tracks2:
                ids_frame_2.add(t["track_id"])

            # In consecutive surveillance video frames, there MUST be overlapping track IDs
            persisting_ids = ids_frame_1.intersection(ids_frame_2)
            self.assertGreater(
                len(persisting_ids),
                0,
                f"Expected persisting IDs between frames, but got Frame1: {ids_frame_1}, Frame2: {ids_frame_2}",
            )
            print(f" [PASS] Test 4: Temporal ID persistence verified! Persisting IDs: {sorted(list(persisting_ids))}")

    def test_05_tracker_reset(self):
        """Test that reset clears active track history."""
        tracker = ByteTracker(model_path=self.MODEL_PATH)
        # Mock active state
        tracker.active_track_ids.add(1)
        tracker.active_track_ids.add(2)
        tracker.total_unique_tracks_seen = 2

        tracker.reset()
        self.assertEqual(len(tracker.active_track_ids), 0)
        self.assertEqual(tracker.total_unique_tracks_seen, 0)
        print(" [PASS] Test 5: Tracker state reset verified")

    def test_06_draw_tracks(self):
        """Test track drawing annotation helper."""
        tracker = ByteTracker(model_path=self.MODEL_PATH)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_tracks = [{
            "track_id": 5,
            "bbox": [100, 100, 200, 300],
            "confidence": 0.89,
            "class_id": 0,
            "class_name": "person",
            "center": (150, 200),
        }]
        annotated = tracker.draw_tracks(dummy_frame, dummy_tracks)
        self.assertEqual(annotated.shape, dummy_frame.shape)
        self.assertGreater(np.sum(annotated), 0)
        print(" [PASS] Test 6: Track annotation drawing verified")

    def test_07_dinesight_calibrated_config(self):
        """Verify that the DineSight custom ByteTrack config file exists and contains valid parameters."""
        cfg_path = "config/bytetrack_dinesight.yaml"
        self.assertTrue(os.path.exists(cfg_path), f"Missing {cfg_path}")
        with open(cfg_path) as fp:
            content = fp.read()
        self.assertIn("tracker_type: bytetrack", content)
        self.assertIn("track_buffer: 60", content)
        self.assertIn("match_thresh: 0.80", content)
        print(" [PASS] Test 7: DineSight custom ByteTrack configuration verified")

    def test_08_dynamic_confidence_and_config_setters(self):
        """Verify dynamic setters for confidence threshold and tracker configuration."""
        tracker = ByteTracker(model_path=self.MODEL_PATH)
        tracker.set_confidence_threshold(0.22)
        self.assertEqual(tracker.conf_threshold, 0.22)

        tracker.set_tracker_type("bytetrack.yaml")
        self.assertEqual(tracker.tracker_type, "bytetrack.yaml")

        with self.assertRaises(ValueError):
            tracker.set_confidence_threshold(-0.1)
        with self.assertRaises(ValueError):
            tracker.set_confidence_threshold(1.2)
        print(" [PASS] Test 8: Dynamic setters and boundary validation verified")

    def test_09_track_metadata_and_duration(self):
        """Verify track metadata tracking and duration calculation."""
        tracker = ByteTracker(model_path=self.MODEL_PATH)
        self.assertEqual(tracker.get_track_duration(999), 0.0)

        # Mock track activity
        tracker.track_metadata[1] = {
            "first_seen": 100.0,
            "last_seen": 105.5,
            "frame_count": 55,
        }
        self.assertAlmostEqual(tracker.get_track_duration(1), 5.5, places=2)

        tracker.reset()
        self.assertEqual(len(tracker.track_metadata), 0)
        self.assertEqual(tracker.get_track_duration(1), 0.0)
        print(" [PASS] Test 9: Track metadata and duration calculation verified")


if __name__ == "__main__":
    print("\nRunning DineSight AI - Phase 2 ByteTracker Test Suite...\n" + "=" * 55)
    unittest.main(verbosity=2)

