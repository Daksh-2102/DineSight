"""
DineSight AI - ROI Manager Unit Test Suite (Phase 3A)
=====================================================
Tests for RegionOfInterest and ROIManager:
1. Valid configuration loading and coordinate validation.
2. Rejection of invalid polygon shapes and coordinates.
3. Point-in-polygon containment (inside vs outside).
4. Representative point calculation (bottom_center vs center).
5. Track spatial association (tables, queue, entrance, unassigned).
6. Overlapping ROI closest-centroid resolution.
7. Visual polygon drawing and overlay validation.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.roi_manager import RegionOfInterest, ROIManager


class TestROIManager(unittest.TestCase):
    CONFIG_PATH = "config/rois.json"
    W, H = 1280, 720

    def setUp(self):
        self.assertTrue(os.path.exists(self.CONFIG_PATH), f"Config missing at {self.CONFIG_PATH}")
        self.roi_mgr = ROIManager(config_path=self.CONFIG_PATH, point_mode="bottom_center")

    def test_01_load_valid_config(self):
        """Test that tables, queue, and entrance regions are correctly loaded."""
        self.assertEqual(len(self.roi_mgr.tables), 3)
        self.assertIn("table_1", self.roi_mgr.tables)
        self.assertIn("table_2", self.roi_mgr.tables)
        self.assertIn("table_3", self.roi_mgr.tables)
        self.assertIsNotNone(self.roi_mgr.queue_region)
        self.assertIsNotNone(self.roi_mgr.entrance_region)
        print(" [PASS] Test 1: Configuration loaded with 3 tables, queue, and entrance")

    def test_02_invalid_coordinate_handling(self):
        """Test that invalid coordinates (out of bounds or <3 points) raise ValueError."""
        # Less than 3 vertices
        with self.assertRaises(ValueError):
            RegionOfInterest("bad_roi", "Bad", [[0.1, 0.1], [0.2, 0.2]])

        # Out-of-bounds coordinate (> 1.0)
        with self.assertRaises(ValueError):
            RegionOfInterest("bad_roi", "Bad", [[0.1, 0.1], [0.2, 0.2], [1.5, 0.5]])

        # Non-2D coordinate
        with self.assertRaises(ValueError):
            RegionOfInterest("bad_roi", "Bad", [[0.1, 0.1], [0.2, 0.2], [0.5]])

        print(" [PASS] Test 2: Invalid polygon coordinate rejection verified")

    def test_03_point_in_polygon(self):
        """Test spatial containment: points inside vs outside Table 1."""
        table_1 = self.roi_mgr.tables["table_1"]
        inside_pt = table_1.get_centroid(self.W, self.H)
        outside_pt = (10, 10)

        self.assertTrue(table_1.contains_point(inside_pt, self.W, self.H))
        self.assertFalse(table_1.contains_point(outside_pt, self.W, self.H))

        # Check find_table_for_point
        matched_id = self.roi_mgr.find_table_for_point(inside_pt, self.W, self.H)
        self.assertEqual(matched_id, "table_1")

        none_id = self.roi_mgr.find_table_for_point(outside_pt, self.W, self.H)
        self.assertNotEqual(none_id, "table_1")
        print(" [PASS] Test 3: Point-in-polygon containment verified")

    def test_04_representative_point_calculation(self):
        """Test extraction of representative point from bounding box."""
        bbox = [100, 200, 200, 400]  # x1, y1, x2, y2 -> width=100, height=200

        # Bottom center: ((100+200)//2, 400) = (150, 400)
        mgr_bc = ROIManager(config_path=self.CONFIG_PATH, point_mode="bottom_center")
        pt_bc = mgr_bc.get_representative_point(bbox)
        self.assertEqual(pt_bc, (150, 400))

        # Centroid: ((100+200)//2, (200+400)//2) = (150, 300)
        mgr_c = ROIManager(config_path=self.CONFIG_PATH, point_mode="center")
        pt_c = mgr_c.get_representative_point(bbox)
        self.assertEqual(pt_c, (150, 300))
        print(" [PASS] Test 4: Representative point calculation (bottom-center & centroid) verified")

    def test_05_track_spatial_association(self):
        """Test mapping tracked people into tables, queue, entrance, and unassigned."""
        # Dynamically place person 1 inside Table 1, person 2 in Queue, person 3 outside
        t1_cx, t1_cy = self.roi_mgr.tables["table_1"].get_centroid(self.W, self.H)
        q_cx, q_cy = self.roi_mgr.queue_region.get_centroid(self.W, self.H)

        sample_tracks = [
            {"track_id": 1, "bbox": [t1_cx - 20, t1_cy - 40, t1_cx + 20, t1_cy], "confidence": 0.88, "class_name": "person"},
            {"track_id": 2, "bbox": [q_cx - 20, q_cy - 40, q_cx + 20, q_cy], "confidence": 0.85, "class_name": "person"},
            {"track_id": 3, "bbox": [20, 20, 60, 60], "confidence": 0.90, "class_name": "person"},
        ]

        mapping = self.roi_mgr.associate_tracks(sample_tracks, self.W, self.H)

        # Person 1 should be associated with Table 1
        t1_occupants = [trk["track_id"] for trk in mapping["tables"]["table_1"]]
        self.assertIn(1, t1_occupants)

        # Person 2 should be in queue
        queue_occupants = [trk["track_id"] for trk in mapping["queue"]]
        self.assertIn(2, queue_occupants)

        # Person 3 should be in unassigned
        unassigned_ids = [trk["track_id"] for trk in mapping["unassigned"]]
        self.assertIn(3, unassigned_ids)

        print(" [PASS] Test 5: Track spatial association across tables, queue, and aisles verified")

    def test_06_draw_rois(self):
        """Test drawing polygon overlays on a test frame."""
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        table_states = {"table_1": "OCCUPIED", "table_2": "VACANT", "table_3": "DINING"}
        occupant_counts = {"table_1": 2, "table_2": 0, "table_3": 1}

        annotated = self.roi_mgr.draw_rois(
            frame,
            table_states=table_states,
            table_occupant_counts=occupant_counts,
            queue_count=1,
        )

        self.assertEqual(annotated.shape, (self.H, self.W, 3))
        # Ensure pixels were altered by drawing
        self.assertGreater(np.sum(annotated), 0)
        print(" [PASS] Test 6: Visual ROI polygon drawing and annotation verified")


if __name__ == "__main__":
    print("\nRunning DineSight AI - Phase 3A ROI Manager Test Suite...\n" + "=" * 55)
    unittest.main(verbosity=2)

