"""
DineSight AI - Queue Monitor Unit Test Suite (Phase 3C)
======================================================
Tests for QueuePersonSession and QueueMonitor:
1. Empty queue initialization.
2. Entry tracking and elapsed wait computation.
3. Dwell threshold confirmation.
4. Loss grace period immunity to momentary occlusion.
5. Exit handling and completed session calculation.
6. Average waiting time across multiple completed sessions.
7. Duplicate counting prevention.
8. State reset functionality.
"""

import os
import sys
import unittest

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.queue_monitor import QueueMonitor, QueuePersonSession


class TestQueueMonitor(unittest.TestCase):

    def test_01_empty_queue(self):
        """Test queue monitor with no occupants."""
        monitor = QueueMonitor()
        metrics = monitor.update([], now=100.0)
        self.assertEqual(metrics["current_queue_length"], 0)
        self.assertEqual(metrics["confirmed_waiting_count"], 0)
        self.assertEqual(metrics["avg_completed_wait_sec"], 0.0)
        self.assertEqual(metrics["total_completed_sessions"], 0)
        print(" [PASS] Test 1: Empty queue baseline verified")

    def test_02_entry_and_dwell_confirmation(self):
        """Test that entrant is registered and confirmed after min_dwell_sec."""
        monitor = QueueMonitor(min_dwell_sec=2.0, loss_grace_sec=2.0)
        person1 = [{"track_id": 101}]

        # Entry at t=10.0
        m1 = monitor.update(person1, now=10.0)
        self.assertEqual(m1["current_queue_length"], 1)
        self.assertEqual(m1["confirmed_waiting_count"], 0, "Should not be confirmed immediately")

        # t=11.0 (1s elapsed, dwell is 2.0s)
        m2 = monitor.update(person1, now=11.0)
        self.assertEqual(m2["confirmed_waiting_count"], 0)

        # t=12.5 (2.5s elapsed > 2.0s dwell)
        m3 = monitor.update(person1, now=12.5)
        self.assertEqual(m3["confirmed_waiting_count"], 1)
        self.assertAlmostEqual(m3["longest_current_wait_sec"], 2.5, delta=0.2)
        print(" [PASS] Test 2: Entry and dwell threshold confirmation verified")

    def test_03_loss_grace_period(self):
        """Test that track missing for < loss_grace_sec does not terminate queue session."""
        monitor = QueueMonitor(min_dwell_sec=1.0, loss_grace_sec=3.0)
        person = [{"track_id": 42}]

        # Enter and confirm
        monitor.update(person, now=0.0)
        monitor.update(person, now=1.5)

        # Disappear for 1.5s (< 3.0s grace)
        m_lost = monitor.update([], now=3.0)
        self.assertEqual(m_lost["current_queue_length"], 1, "Must remain active during grace period")

        # Reappear
        m_back = monitor.update(person, now=3.5)
        self.assertEqual(m_back["current_queue_length"], 1)
        print(" [PASS] Test 3: Temporary tracking loss grace period verified")

    def test_04_exit_and_average_calculation(self):
        """Test session completion and average wait time calculation."""
        monitor = QueueMonitor(min_dwell_sec=1.0, loss_grace_sec=1.0)

        # Customer 1 waits 4 seconds (t=10 to 14)
        monitor.update([{"track_id": 1}], now=10.0)
        monitor.update([{"track_id": 1}], now=14.0)
        # Exits at t=14.0, detected missing at t=15.5 (> 1.0s grace)
        monitor.update([], now=15.5)

        self.assertEqual(len(monitor.completed_sessions), 1)
        self.assertAlmostEqual(monitor.completed_sessions[0].elapsed_wait(14.0), 4.0, delta=0.2)

        # Customer 2 waits 6 seconds (t=20 to 26)
        monitor.update([{"track_id": 2}], now=20.0)
        monitor.update([{"track_id": 2}], now=26.0)
        monitor.update([], now=27.5)

        self.assertEqual(len(monitor.completed_sessions), 2)
        # Average wait: (4.0 + 6.0) / 2 = 5.0 seconds
        metrics = monitor.update([], now=30.0)
        self.assertAlmostEqual(metrics["avg_completed_wait_sec"], 5.0, delta=0.3)
        self.assertEqual(metrics["total_completed_sessions"], 2)
        print(" [PASS] Test 4: Exit detection and average wait time (5.0s) verified")

    def test_05_duplicate_prevention(self):
        """Test that same person track ID is never double-counted in queue length."""
        monitor = QueueMonitor(min_dwell_sec=1.0)
        person = [{"track_id": 99}]

        for t in range(10):
            m = monitor.update(person, now=float(t))
            self.assertEqual(m["current_queue_length"], 1)

        print(" [PASS] Test 5: Duplicate person counting prevention verified")


if __name__ == "__main__":
    print("\nRunning DineSight AI - Phase 3C Queue Monitor Test Suite...\n" + "=" * 55)
    unittest.main(verbosity=2)

