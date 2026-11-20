"""
DineSight AI - Table State Engine Unit Test Suite (Phase 3B)
============================================================
Tests for TableSession and TableStateEngine:
1. Initial state is VACANT.
2. Dwell-time requirement (passers-by do not trigger occupancy).
3. Occupancy confirmation after dwell threshold.
4. Release grace period (temporary occlusion does not drop occupancy).
5. Vacant transition after release duration.
6. Activity state estimation (WAITING_FOR_FOOD vs DINING).
7. Reset functionality.
"""

import os
import sys
import unittest

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.state_engine import TableSession, TableState, TableStateEngine


class TestTableStateEngine(unittest.TestCase):

    def test_01_initial_vacant(self):
        """Test that new tables start in VACANT state."""
        session = TableSession("table_1", dwell_time_sec=2.0, release_time_sec=2.0)
        self.assertEqual(session.current_state, TableState.VACANT)
        status = session.get_status(now=100.0)
        self.assertEqual(status["state"], "VACANT")
        self.assertFalse(status["is_occupied"])
        print(" [PASS] Test 1: Initial VACANT state verified")

    def test_02_dwell_time_rejection_for_passers_by(self):
        """Test that a person walking through for < 2.0s does NOT trigger OCCUPIED."""
        session = TableSession("table_1", dwell_time_sec=2.0, release_time_sec=2.0)
        sample_occupant = [{"track_id": 10}]

        # t=100.0s: Person steps in
        session.update(sample_occupant, now=100.0)
        self.assertEqual(session.current_state, TableState.VACANT)

        # t=101.0s: 1 second elapsed (dwell is 2.0s) -> STILL VACANT
        session.update(sample_occupant, now=101.0)
        self.assertEqual(session.current_state, TableState.VACANT)

        # t=101.5s: Person leaves before dwell expired
        session.update([], now=101.5)
        self.assertEqual(session.current_state, TableState.VACANT)

        print(" [PASS] Test 2: Dwell-time rejection of brief passers-by verified")

    def test_03_occupancy_confirmation_after_dwell(self):
        """Test that person staying >= dwell_time triggers OCCUPIED / WAITING_FOR_FOOD."""
        session = TableSession("table_1", dwell_time_sec=2.0, release_time_sec=2.0)
        sample_occupant = [{"track_id": 5}]

        session.update(sample_occupant, now=100.0)
        session.update(sample_occupant, now=101.0)
        # t=102.1s: Dwell time satisfied!
        session.update(sample_occupant, now=102.1)

        self.assertIn(session.current_state, (TableState.OCCUPIED, TableState.WAITING_FOR_FOOD))
        status = session.get_status(now=102.1)
        self.assertTrue(status["is_occupied"])
        self.assertIn(5, status["occupant_ids"])
        print(" [PASS] Test 3: Occupancy confirmation after dwell threshold verified")

    def test_04_release_time_grace_period(self):
        """Test that momentary occlusion does not prematurely revert table to VACANT."""
        session = TableSession("table_1", dwell_time_sec=1.0, release_time_sec=3.0)
        occupant = [{"track_id": 1}]

        # Establish occupancy
        session.update(occupant, now=10.0)
        session.update(occupant, now=11.5)  # Occupied!
        self.assertTrue(session.current_state != TableState.VACANT)

        # Occlusion for 1.5s (less than 3.0s release)
        session.update([], now=12.0)
        session.update([], now=13.0)
        self.assertTrue(session.current_state != TableState.VACANT, "Table must stay occupied during grace period")

        # Person reappears at 13.5s
        session.update(occupant, now=13.5)
        self.assertTrue(session.current_state != TableState.VACANT)
        print(" [PASS] Test 4: Release grace period / temporary occlusion immunity verified")

    def test_05_vacant_after_full_release(self):
        """Test that table returns to VACANT once release time has fully expired."""
        session = TableSession("table_1", dwell_time_sec=1.0, release_time_sec=2.0)
        occupant = [{"track_id": 1}]

        # Establish occupancy
        session.update(occupant, now=10.0)
        session.update(occupant, now=11.2)

        # Person leaves at 12.0s
        session.update([], now=12.0)
        session.update([], now=13.0)
        # At 14.5s: 2.5s > 2.0s release
        session.update([], now=14.5)
        self.assertEqual(session.current_state, TableState.VACANT)
        print(" [PASS] Test 5: Table cleanly returns to VACANT after patron departure")

    def test_06_activity_state_dining_vs_waiting(self):
        """Test transition between WAITING_FOR_FOOD and DINING based on tableware evidence."""
        session = TableSession(
            "table_1",
            dwell_time_sec=1.0,
            waiting_threshold_sec=0.5,
            activity_smoothing_sec=1.0,
        )
        occupant = [{"track_id": 7}]
        tableware = [{"class_name": "cup"}]

        # Occupy table without tableware for > dwell + waiting_threshold (1.6s >= 1.5s)
        session.update(occupant, tableware_detections=[], now=10.0)
        session.update(occupant, tableware_detections=[], now=11.6)
        self.assertEqual(session.current_state, TableState.WAITING_FOR_FOOD)

        # Tableware arrives and persists
        session.update(occupant, tableware_detections=tableware, now=12.0)
        session.update(occupant, tableware_detections=tableware, now=13.2)
        self.assertEqual(session.current_state, TableState.DINING)
        print(" [PASS] Test 6: Activity state transitions (WAITING_FOR_FOOD -> DINING) verified")

    def test_07_grounded_primary_state_before_threshold(self):
        """Test that newly occupied table stays OCCUPIED (grounded) before waiting threshold expires."""
        session = TableSession(
            "table_1",
            dwell_time_sec=1.0,
            waiting_threshold_sec=10.0,
            activity_smoothing_sec=1.0,
        )
        occupant = [{"track_id": 9}]

        # Dwell satisfied at t=11.5s
        session.update(occupant, tableware_detections=[], now=10.0)
        session.update(occupant, tableware_detections=[], now=11.5)

        # Should be OCCUPIED, NOT WAITING_FOR_FOOD yet (only 1.5s seated, threshold is 10s)
        self.assertEqual(session.primary_state.value, "OCCUPIED")
        self.assertEqual(session.current_state, TableState.OCCUPIED)
        self.assertEqual(session.activity_state.value, "UNKNOWN")

        # After 12 seconds (t=22.0s), waiting threshold expires -> transitions to WAITING_FOR_FOOD
        session.update(occupant, tableware_detections=[], now=22.0)
        self.assertEqual(session.current_state, TableState.WAITING_FOR_FOOD)
        self.assertEqual(session.activity_state.value, "ESTIMATED_WAITING")

        print(" [PASS] Test 7: Grounded OCCUPIED state and non-presumptive transition verified")

    def test_08_structured_status_metadata(self):
        """Test that get_status() returns decoupled primary_state, activity_state, and duration."""
        session = TableSession("table_2", dwell_time_sec=1.0)
        occupant = [{"track_id": 12}]

        session.update(occupant, now=10.0)
        session.update(occupant, now=11.5)

        status = session.get_status(now=15.0)
        self.assertIn("primary_state", status)
        self.assertIn("activity_state", status)
        self.assertIn("occupied_duration_sec", status)
        self.assertEqual(status["primary_state"], "OCCUPIED")
        self.assertTrue(status["is_occupied"])
        self.assertAlmostEqual(status["occupied_duration_sec"], 3.5, places=1)

        print(" [PASS] Test 8: Structured status metadata verified")

    def test_09_vacant_table_with_leftover_tableware_stays_vacant(self):
        """Test that an empty table with leftover dirty dishes does NOT trigger OCCUPIED or DINING."""
        session = TableSession("table_3", dwell_time_sec=1.0)
        tableware = [{"class_name": "bottle"}, {"class_name": "cup"}]

        # Tableware present, but 0 people seated
        st1 = session.update(occupant_tracks=[], tableware_detections=tableware, now=10.0)
        st2 = session.update(occupant_tracks=[], tableware_detections=tableware, now=15.0)

        self.assertEqual(st1, TableState.VACANT)
        self.assertEqual(st2, TableState.VACANT)
        self.assertEqual(session.primary_state.value, "VACANT")
        self.assertFalse(session.get_status()["is_occupied"])
        print(" [PASS] Test 9: Vacant table with leftover tableware stays VACANT verified")


if __name__ == "__main__":
    print("\nRunning DineSight AI - Phase 3B Table State Engine Test Suite...\n" + "=" * 55)
    unittest.main(verbosity=2)

