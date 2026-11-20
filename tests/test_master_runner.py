"""
DineSight AI - Master Runner & Multi-Phase Integration Tests (Milestone 12)
===========================================================================
Validates end-to-end integration across all runner scripts and operational modes:
1. Master Application Runner (run.py) in RAW, DETECT, TRACK, and FULL modes.
2. Command-line argument parsing and parameter overrides.
3. Phase-specific launcher scripts (run_phase1.py, run_phase2.py, run_phase3.py).
4. Headless execution stability and graceful resource deallocation.
"""

import os
import sys
import subprocess
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestMasterRunnerIntegration(unittest.TestCase):
    """End-to-end integration test suite for DineSight runners and CLI interfaces."""

    @classmethod
    def setUpClass(cls):
        cls.python_exe = sys.executable
        cls.video_path = "data/sample_videos/restaurant_cctv_sample.mp4"
        cls.assertTrue(os.path.exists(cls.video_path), f"Video not found: {cls.video_path}")

    def _run_script(self, script_path: str, extra_args: list) -> subprocess.CompletedProcess:
        """Helper to invoke a Python script with timeout and output capture."""
        cmd = [self.python_exe, script_path, "--source", self.video_path, "--headless", "--max-frames", "5"] + extra_args
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        return result

    def test_01_master_runner_raw_mode(self):
        """Verify run.py executes cleanly in RAW mode."""
        res = self._run_script("run.py", ["--mode", "raw"])
        self.assertEqual(res.returncode, 0, f"run.py RAW failed with stderr:\n{res.stderr}")
        self.assertIn("Active Mode  : RAW", res.stdout)
        self.assertIn("Reached target max frames", res.stdout)

    def test_02_master_runner_detect_mode(self):
        """Verify run.py executes cleanly in DETECT mode."""
        res = self._run_script("run.py", ["--mode", "detect", "--conf", "0.25"])
        self.assertEqual(res.returncode, 0, f"run.py DETECT failed with stderr:\n{res.stderr}")
        self.assertIn("Active Mode  : DETECT", res.stdout)
        self.assertIn("Reached target max frames", res.stdout)

    def test_03_master_runner_track_mode(self):
        """Verify run.py executes cleanly in TRACK mode."""
        res = self._run_script("run.py", ["--mode", "track", "--conf", "0.20"])
        self.assertEqual(res.returncode, 0, f"run.py TRACK failed with stderr:\n{res.stderr}")
        self.assertIn("Active Mode  : TRACK", res.stdout)
        self.assertIn("Reached target max frames", res.stdout)

    def test_04_master_runner_full_mode(self):
        """Verify run.py executes cleanly in FULL mode with hybrid spatial association."""
        res = self._run_script("run.py", ["--mode", "full", "--assoc-mode", "hybrid", "--tableware"])
        self.assertEqual(res.returncode, 0, f"run.py FULL failed with stderr:\n{res.stderr}")
        self.assertIn("Active Mode  : FULL", res.stdout)
        self.assertIn("Reached target max frames", res.stdout)

    def test_05_phase1_runner(self):
        """Verify run_phase1.py executes cleanly in headless mode."""
        res = self._run_script("run_phase1.py", [])
        self.assertEqual(res.returncode, 0, f"run_phase1.py failed with stderr:\n{res.stderr}")
        self.assertIn("VIDEO INGESTION ENGINE", res.stdout)
        self.assertIn("Reached target max frames", res.stdout)

    def test_06_phase2_runner(self):
        """Verify run_phase2.py executes cleanly in headless mode."""
        res = self._run_script("run_phase2.py", ["--conf", "0.25"])
        self.assertEqual(res.returncode, 0, f"run_phase2.py failed with stderr:\n{res.stderr}")
        self.assertIn("PHASE 2: DETECTION & TRACKING", res.stdout)
        self.assertIn("Reached target max frames", res.stdout)

    def test_07_phase3_runner(self):
        """Verify run_phase3.py executes cleanly in headless mode."""
        res = self._run_script("run_phase3.py", ["--dwell", "2.0"])
        self.assertEqual(res.returncode, 0, f"run_phase3.py failed with stderr:\n{res.stderr}")
        self.assertIn("PHASE 3: TABLE STATES & QUEUE MONITORING", res.stdout)
        self.assertIn("Reached target max frames", res.stdout)


if __name__ == "__main__":
    unittest.main()
