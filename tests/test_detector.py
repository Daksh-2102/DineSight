"""
DineSight AI - YOLOv8 Detector Test Suite (Phase 2)
===================================================
Tests for YOLODetector:
1. Model loading and class verification.
2. Device resolution (CPU / CUDA).
3. Handling of blank / empty frames without detections.
4. Detection output structure (Integration test on sample video).
5. Person-only class filtering.
6. Graceful error handling on missing weights.
7. Visual bounding box annotation.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.detector import YOLODetector
from backend.app.core.video_reader import VideoReader


class TestYOLODetector(unittest.TestCase):
    MODEL_PATH = "data/models/yolov8n.pt"
    SAMPLE_VIDEO = "data/sample_videos/pedestrians_surveillance.mp4"

    def setUp(self):
        self.assertTrue(
            os.path.exists(self.MODEL_PATH),
            f"Model weights not found at {self.MODEL_PATH}. Run Phase 2 setup first.",
        )

    def test_01_model_loading(self):
        """Test that YOLOv8 Nano loads and class dictionary is populated."""
        detector = YOLODetector(model_path=self.MODEL_PATH)
        self.assertIsNotNone(detector.model)
        self.assertIn(0, detector.class_names)
        self.assertEqual(detector.class_names[0], "person")
        print(" [PASS] Test 1: YOLOv8n model loading & class dictionary verification")

    def test_02_device_resolution(self):
        """Test device resolution for CPU and auto."""
        detector_cpu = YOLODetector(model_path=self.MODEL_PATH, device="cpu")
        self.assertEqual(detector_cpu.device, "cpu")
        self.assertEqual(detector_cpu.device_name, "CPU")
        print(" [PASS] Test 2: Hardware device resolution (CPU verified)")

    def test_03_empty_frame_handling(self):
        """Test that a blank (black) frame returns an empty list without crashing."""
        detector = YOLODetector(model_path=self.MODEL_PATH, conf_threshold=0.5)
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(black_frame)
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 0)
        print(" [PASS] Test 3: Blank frame returns empty detection list gracefully")

    def test_04_detection_structure(self):
        """[Integration Test] Verify detection output format on real video frame."""
        self.assertTrue(os.path.exists(self.SAMPLE_VIDEO), "Sample video missing")
        with VideoReader(self.SAMPLE_VIDEO) as reader:
            success, frame, _ = reader.read_frame()
            self.assertTrue(success)

            detector = YOLODetector(model_path=self.MODEL_PATH, conf_threshold=0.35)
            detections = detector.detect(frame)

            self.assertIsInstance(detections, list)
            self.assertGreater(len(detections), 0, "Expected at least 1 person detection in sample video")

            # Check structure of the first detection
            det = detections[0]
            for key in ["bbox", "confidence", "class_id", "class_name", "center"]:
                self.assertIn(key, det)

            x1, y1, x2, y2 = det["bbox"]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(det["confidence"], 0.35)
            self.assertIsInstance(det["center"], tuple)
            self.assertEqual(len(det["center"]), 2)

            print(f" [PASS] Test 4: Detection structure verified ({len(detections)} people detected in frame)")

    def test_05_person_class_filtering(self):
        """Verify that only the requested target class (0: person) is returned."""
        with VideoReader(self.SAMPLE_VIDEO) as reader:
            success, frame, _ = reader.read_frame()
            detector = YOLODetector(model_path=self.MODEL_PATH, target_classes=[0], conf_threshold=0.3)
            detections = detector.detect(frame)

            for det in detections:
                self.assertEqual(det["class_id"], 0)
                self.assertEqual(det["class_name"], "person")

            print(f" [PASS] Test 5: Person-class filtering strictly verified on {len(detections)} objects")

    def test_06_missing_weights_error(self):
        """Verify that an invalid weights file path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            YOLODetector(model_path="data/models/non_existent_weights.pt")
        print(" [PASS] Test 6: Missing weights error handling verified")

    def test_07_draw_detections(self):
        """Verify that bounding box annotation helper returns an annotated frame."""
        detector = YOLODetector(model_path=self.MODEL_PATH)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_dets = [{
            "bbox": [50, 50, 150, 200],
            "confidence": 0.88,
            "class_id": 0,
            "class_name": "person",
            "center": (100, 125),
        }]
        annotated = detector.draw_detections(dummy_frame, dummy_dets)
        self.assertEqual(annotated.shape, dummy_frame.shape)
        # Verify pixels inside badge/box were modified
        self.assertGreater(np.sum(annotated), 0)
        print(" [PASS] Test 7: Bounding box drawing verified")


if __name__ == "__main__":
    print("\nRunning DineSight AI - Phase 2 YOLODetector Test Suite...\n" + "=" * 55)
    unittest.main(verbosity=2)

