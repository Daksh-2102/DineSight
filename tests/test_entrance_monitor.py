"""
DineSight AI - Entrance & Customer Footfall Monitor Unit Test Suite (Milestone 8)
================================================================================
Comprehensive verification for EntranceMonitor:
1. Empty baseline (0 entries, 0 exits, 0 occupancy).
2. Clean entrance crossing (Exterior -> Interior) triggers ENTRY.
3. Clean exit crossing (Interior -> Exterior) triggers EXIT.
4. Debounce and loitering immunity (no duplicate counting on hesitation).
5. Lateral containment (people walking in far aisles do not trigger doorway count).
6. Spatial hysteresis buffer (jitter within buffer does not flicker events).
7. Multiple concurrent patrons (net occupancy arithmetic).
8. Dynamic tripwire reconfiguration.
9. State reset functionality.
10. Visual rendering on frame canvas.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.entrance_monitor import EntranceMonitor, FootfallEvent


class TestEntranceMonitor(unittest.TestCase):

    def setUp(self):
        # Configure standard doorway tripwire: horizontal segment from (100, 200) to (300, 200)
        # on an 800x600 canvas. Exterior is y < 200, Interior is y > 200.
        self.tripwire = [[100, 200], [300, 200]]
        self.monitor = EntranceMonitor(
            tripwire_line=self.tripwire,
            hysteresis_band_px=15.0,
            lateral_margin=0.15,
            debounce_cooldown_sec=3.0,
            loss_grace_sec=2.0,
        )

    def test_01_empty_baseline(self):
        """Test initial state with no patrons."""
        res = self.monitor.update([], frame_width=800, frame_height=600, now=100.0)
        self.assertEqual(res["total_entries"], 0)
        self.assertEqual(res["total_exits"], 0)
        self.assertEqual(res["net_occupants"], 0)
        self.assertEqual(res["new_entries"], 0)
        self.assertEqual(res["new_exits"], 0)
        print(" [PASS] Test 1: Empty entrance baseline verified")

    def test_02_clean_entry_crossing(self):
        """Test a patron moving from exterior (y=160) into interior (y=240)."""
        # Step 1: Person outside at (200, 160) [Exterior, dist = -40]
        trk_t1 = [{"track_id": 1, "bbox": [180, 100, 220, 160]}]
        r1 = self.monitor.update(trk_t1, frame_width=800, frame_height=600, now=10.0)
        self.assertEqual(r1["total_entries"], 0)

        # Step 2: Person crosses into interior at (200, 240) [Interior, dist = +40]
        trk_t2 = [{"track_id": 1, "bbox": [180, 180, 220, 240]}]
        r2 = self.monitor.update(trk_t2, frame_width=800, frame_height=600, now=11.0)
        self.assertEqual(r2["total_entries"], 1)
        self.assertEqual(r2["total_exits"], 0)
        self.assertEqual(r2["net_occupants"], 1)
        self.assertEqual(r2["new_entries"], 1)
        self.assertEqual(len(self.monitor.completed_events), 1)
        self.assertEqual(self.monitor.completed_events[0].direction, "ENTRY")
        print(" [PASS] Test 2: Clean directional ENTRY crossing verified")

    def test_03_clean_exit_crossing(self):
        """Test a patron moving from dining room interior (y=250) out to exterior (y=150)."""
        # Step 1: Person inside at (200, 250) [Interior, dist = +50]
        trk_t1 = [{"track_id": 2, "bbox": [180, 190, 220, 250]}]
        r1 = self.monitor.update(trk_t1, frame_width=800, frame_height=600, now=20.0)
        self.assertEqual(r1["total_exits"], 0)

        # Step 2: Person exits to (200, 150) [Exterior, dist = -50]
        trk_t2 = [{"track_id": 2, "bbox": [180, 90, 220, 150]}]
        r2 = self.monitor.update(trk_t2, frame_width=800, frame_height=600, now=21.0)
        self.assertEqual(r2["total_exits"], 1)
        self.assertEqual(r2["total_entries"], 0)
        self.assertEqual(r2["net_occupants"], 0)
        self.assertEqual(r2["new_exits"], 1)
        self.assertEqual(len(self.monitor.completed_events), 1)
        self.assertEqual(self.monitor.completed_events[0].direction, "EXIT")
        print(" [PASS] Test 3: Clean directional EXIT crossing verified")

    def test_04_debounce_and_loitering_immunity(self):
        """Test that lingering near the doorway within cooldown does not double-count."""
        # Initial entry
        t1 = [{"track_id": 5, "bbox": [190, 110, 210, 160]}]  # y=160 (exterior)
        t2 = [{"track_id": 5, "bbox": [190, 190, 210, 240]}]  # y=240 (interior)
        self.monitor.update(t1, 800, 600, now=30.0)
        res = self.monitor.update(t2, 800, 600, now=31.0)
        self.assertEqual(res["total_entries"], 1)

        # Person steps back slightly to y=230 (still interior), then y=245 at t=32.0 (1.0s elapsed < 3.0s cooldown)
        t3 = [{"track_id": 5, "bbox": [190, 180, 210, 230]}]
        t4 = [{"track_id": 5, "bbox": [190, 195, 210, 245]}]
        self.monitor.update(t3, 800, 600, now=31.5)
        res4 = self.monitor.update(t4, 800, 600, now=32.0)
        self.assertEqual(res4["total_entries"], 1, "Duplicate entry must be suppressed by debounce")
        self.assertEqual(res4["new_entries"], 0)
        print(" [PASS] Test 4: Debounce and loitering immunity verified")

    def test_05_lateral_bounds_containment(self):
        """Test that a person moving far off to the side (x=500) does not trigger doorway crossing."""
        # Person moves from y=160 to y=240, but at x=550 (far from tripwire x: 100..300)
        trk1 = [{"track_id": 9, "bbox": [530, 100, 570, 160]}]  # cx=550, cy=160
        trk2 = [{"track_id": 9, "bbox": [530, 180, 570, 240]}]  # cx=550, cy=240
        self.monitor.update(trk1, 800, 600, now=40.0)
        res = self.monitor.update(trk2, 800, 600, now=41.0)

        self.assertEqual(res["total_entries"], 0, "Aisle walker must not trigger entrance event")
        self.assertEqual(res["total_exits"], 0)
        print(" [PASS] Test 5: Lateral bounds containment verified")

    def test_06_hysteresis_buffer_no_jitter_trigger(self):
        """Test that minor bounding box sway within hysteresis buffer [-15, +15] does not trigger."""
        # Line is at y=200. Buffer is [185, 215].
        # Person starts outside at y=150
        trk1 = [{"track_id": 12, "bbox": [190, 100, 210, 150]}]
        self.monitor.update(trk1, 800, 600, now=50.0)

        # Patron steps slightly near the door at y=195 (inside buffer, dist = -5)
        trk2 = [{"track_id": 12, "bbox": [190, 145, 210, 195]}]
        r2 = self.monitor.update(trk2, 800, 600, now=51.0)
        self.assertEqual(r2["total_entries"], 0)

        # Patron sways to y=205 (inside buffer, dist = +5)
        trk3 = [{"track_id": 12, "bbox": [190, 155, 210, 205]}]
        r3 = self.monitor.update(trk3, 800, 600, now=52.0)
        self.assertEqual(r3["total_entries"], 0, "Buffer transition must not trigger until crossing is decisive")

        # Patron retreats back to y=150 (turns around, leaves)
        trk4 = [{"track_id": 12, "bbox": [190, 100, 210, 150]}]
        r4 = self.monitor.update(trk4, 800, 600, now=53.0)
        self.assertEqual(r4["total_entries"], 0)
        self.assertEqual(r4["total_exits"], 0)
        print(" [PASS] Test 6: Spatial hysteresis buffer immunity to jitter verified")

    def test_07_multiple_concurrent_patrons(self):
        """Test concurrent patrons entering and exiting maintains accurate net occupancy."""
        # Patron 10 enters
        self.monitor.update([{"track_id": 10, "bbox": [190, 100, 210, 160]}], 800, 600, now=60.0)
        self.monitor.update([{"track_id": 10, "bbox": [190, 180, 210, 240]}], 800, 600, now=61.0)

        # Patron 20 enters
        self.monitor.update([{"track_id": 20, "bbox": [240, 100, 260, 160]}], 800, 600, now=62.0)
        r = self.monitor.update([{"track_id": 20, "bbox": [240, 180, 260, 240]}], 800, 600, now=63.0)
        self.assertEqual(r["total_entries"], 2)
        self.assertEqual(r["net_occupants"], 2)

        # Patron 10 exits
        self.monitor.update([{"track_id": 10, "bbox": [190, 100, 210, 150]}], 800, 600, now=68.0)
        r_exit = self.monitor.update([], 800, 600, now=69.0)
        self.assertEqual(r_exit["total_entries"], 2)
        self.assertEqual(r_exit["total_exits"], 1)
        self.assertEqual(r_exit["net_occupants"], 1)
        print(" [PASS] Test 7: Multiple concurrent patrons and net occupancy verified")

    def test_08_dynamic_tripwire_reconfiguration(self):
        """Test dynamic repositioning of the tripwire."""
        # Move tripwire to y=400
        self.monitor.set_tripwire((100, 400), (300, 400))
        p1, p2 = self.monitor.get_pixel_tripwire(800, 600)
        self.assertEqual(p1, (100, 400))
        self.assertEqual(p2, (300, 400))

        # Test crossing across new line at y=400
        self.monitor.update([{"track_id": 30, "bbox": [200, 300, 220, 360]}], 800, 600, now=70.0)
        r = self.monitor.update([{"track_id": 30, "bbox": [200, 380, 220, 440]}], 800, 600, now=71.0)
        self.assertEqual(r["total_entries"], 1)
        print(" [PASS] Test 8: Dynamic tripwire reconfiguration verified")

    def test_09_state_reset(self):
        """Test reset cleanly wipes history and counts."""
        # Create an entry
        self.monitor.update([{"track_id": 1, "bbox": [180, 100, 220, 160]}], 800, 600, now=1.0)
        self.monitor.update([{"track_id": 1, "bbox": [180, 180, 220, 240]}], 800, 600, now=2.0)
        self.assertEqual(self.monitor.total_entries, 1)

        self.monitor.reset()
        self.assertEqual(self.monitor.total_entries, 0)
        self.assertEqual(self.monitor.total_exits, 0)
        self.assertEqual(self.monitor.net_occupancy, 0)
        self.assertEqual(len(self.monitor.active_tracks), 0)
        self.assertEqual(len(self.monitor.completed_events), 0)
        print(" [PASS] Test 9: State reset verified")

    def test_10_drawing_visuals(self):
        """Test visual rendering function."""
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        # Add some active tracks
        self.monitor.update([{"track_id": 1, "bbox": [180, 180, 220, 240]}], 800, 600, now=1.0)
        annotated = self.monitor.draw_entrance_visuals(canvas)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, (600, 800, 3))
        # Ensure pixels were drawn (not all zeros)
        self.assertGreater(np.count_nonzero(annotated), 0)
        print(" [PASS] Test 10: Visual rendering function verified")


if __name__ == "__main__":
    unittest.main()
