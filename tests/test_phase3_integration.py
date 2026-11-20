"""
DineSight AI - Phase 3 Integration Test Suite
=============================================
Tests the complete end-to-end integration of:
VideoReader -> ByteTracker -> ROIManager -> TableStateEngine -> QueueMonitor
using real surveillance video frames.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.video_reader import VideoReader
from backend.app.core.tracker import ByteTracker
from backend.app.core.roi_manager import ROIManager
from backend.app.core.state_engine import TableStateEngine, TableState
from backend.app.core.queue_monitor import QueueMonitor


class TestPhase3Integration(unittest.TestCase):
    SAMPLE_VIDEO = "data/sample_videos/pedestrians_surveillance.mp4"
    CONFIG_PATH = "config/rois.json"
    MODEL_PATH = "data/models/yolov8n.pt"

    def setUp(self):
        self.assertTrue(os.path.exists(self.SAMPLE_VIDEO), "Sample video missing")
        self.assertTrue(os.path.exists(self.CONFIG_PATH), "ROI config missing")
        self.assertTrue(os.path.exists(self.MODEL_PATH), "YOLO weights missing")

    def test_01_end_to_end_pipeline(self):
        """Test full pipeline execution across 10 consecutive video frames."""
        reader = VideoReader(self.SAMPLE_VIDEO, target_size=(1280, 720), frame_skip=1)
        roi_mgr = ROIManager(config_path=self.CONFIG_PATH)
        tracker = ByteTracker(model_path=self.MODEL_PATH, conf_threshold=0.35)
        table_ids = list(roi_mgr.tables.keys())
        state_engine = TableStateEngine(table_ids=table_ids, dwell_time_sec=1.0, release_time_sec=2.0)
        queue_monitor = QueueMonitor(min_dwell_sec=1.0)

        processed = 0
        total_tracked_people_seen = 0

        for frame_idx, frame, fps in reader.frames():
            processed += 1
            h, w = frame.shape[:2]

            # 1. ByteTrack
            tracks = tracker.update(frame)
            total_tracked_people_seen += len(tracks)

            # 2. Spatial Mapping
            spatial = roi_mgr.associate_tracks(tracks, frame_width=w, frame_height=h)
            self.assertIn("tables", spatial)
            self.assertIn("queue", spatial)
            self.assertIn("entrance", spatial)
            self.assertIn("unassigned", spatial)

            # 3. Table State Engine
            states = state_engine.update(spatial["tables"])
            self.assertEqual(len(states), len(table_ids))
            for t_id, st in states.items():
                self.assertIn(st, ["VACANT", "OCCUPIED", "WAITING_FOR_FOOD", "DINING"])

            # 4. Queue Monitor
            q_metrics = queue_monitor.update(spatial["queue"])
            self.assertIn("current_queue_length", q_metrics)
            self.assertIn("longest_current_wait_sec", q_metrics)
            self.assertIn("avg_completed_wait_sec", q_metrics)

            # 5. Drawing Verification
            table_counts = {t_id: len(spatial["tables"].get(t_id, [])) for t_id in table_ids}
            annotated = roi_mgr.draw_rois(
                frame.copy(),
                table_states=states,
                table_occupant_counts=table_counts,
                queue_count=q_metrics["current_queue_length"],
            )
            annotated = tracker.draw_tracks(annotated, tracks)
            self.assertEqual(annotated.shape, (720, 1280, 3))

            if processed >= 10:
                break

        reader.release()
        self.assertEqual(processed, 10)
        self.assertGreater(total_tracked_people_seen, 0, "Expected at least 1 person to be tracked across 10 frames")
        print(f" [PASS] Test 1: Complete Phase 3 end-to-end pipeline executed successfully over {processed} frames!")

    def test_02_master_hud_rendering(self):
        """Test that redesigned master HUD renders cleanly across resolutions and modes."""
        import run

        resolutions = [(848, 480), (1280, 720)]
        modes = ["raw", "detect", "track", "full"]

        for w, h in resolutions:
            for mode in modes:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                run.draw_master_hud(
                    frame=canvas,
                    mode=mode,
                    frame_idx=50,
                    total_frames=1000,
                    pipeline_fps=24.5,
                    inf_time_ms=32.0,
                    device_name="CPU",
                    count=4,
                    total_unique=8,
                    table_states={"table_1": "OCCUPIED", "table_2": "DINING"},
                    queue_metrics={"current_queue_length": 1},
                    entrance_metrics={"total_entries": 3, "total_exits": 1, "net_occupants": 2},
                    frame_skip=1,
                    paused=False,
                )
                self.assertEqual(canvas.shape, (h, w, 3))
                self.assertGreater(np.sum(canvas), 0, f"HUD drawing failed for mode {mode} at {w}x{h}")

        print(" [PASS] Test 2: Master HUD multi-resolution and multi-mode rendering verified")


if __name__ == "__main__":
    print("\nRunning DineSight AI - Phase 3 Integration Test Suite...\n" + "=" * 55)
    unittest.main(verbosity=2)

