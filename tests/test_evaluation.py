"""
Tests for Milestone 11: Real CCTV Evaluation & Validation Benchmark
===================================================================
Verifies the integrity of ground truth annotations, evaluation metric
computations (MAE, RMSE, accuracy), benchmark initialization, and reporting.
"""

import os
import sys
import json
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.evaluate_accuracy import CCTVBenchmark, format_markdown_report


class TestEvaluationBenchmark(unittest.TestCase):
    """Test suite for evaluation harness, ground truth, and metric computations."""

    @classmethod
    def setUpClass(cls):
        cls.gt_path = "eval/ground_truth_keyframes.json"
        cls.video_path = "data/sample_videos/restaurant_cctv_sample.mp4"
        cls.roi_path = "config/rois.json"

    def test_01_ground_truth_schema(self):
        """Verify ground_truth_keyframes.json exists and conforms to expected schema."""
        self.assertTrue(os.path.exists(self.gt_path), f"Missing {self.gt_path}")

        with open(self.gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        self.assertIn("dataset", gt)
        self.assertIn("fps", gt)
        self.assertIn("keyframes", gt)
        self.assertGreaterEqual(len(gt["keyframes"]), 5)

        for kf in gt["keyframes"]:
            self.assertIn("frame_index", kf)
            self.assertIn("time_sec", kf)
            self.assertIn("expected_persons", kf)
            self.assertIn("expected_table_states", kf)
            self.assertIsInstance(kf["frame_index"], int)
            self.assertIsInstance(kf["expected_persons"], int)
            self.assertIsInstance(kf["expected_table_states"], dict)

    def test_02_benchmark_initialization(self):
        """Verify CCTVBenchmark initializes all core engines cleanly."""
        benchmark = CCTVBenchmark(
            video_path=self.video_path,
            ground_truth_path=self.gt_path,
            roi_config_path=self.roi_path,
            conf_threshold=0.20,
            iou_threshold=0.50,
        )
        self.assertIsNotNone(benchmark.detector)
        self.assertIsNotNone(benchmark.tracker)
        self.assertIsNotNone(benchmark.roi_mgr)
        self.assertIsNotNone(benchmark.state_engine)
        self.assertIsNotNone(benchmark.queue_monitor)
        self.assertIsNotNone(benchmark.entrance_monitor)

    def test_03_metric_math(self):
        """Verify accuracy metric formulas (MAE, RMSE, accuracy percentage)."""
        actuals = [5, 6, 7, 6, 5]
        predictions = [5, 7, 6, 6, 4]
        errors = [abs(p - a) for p, a in zip(predictions, actuals)]

        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(np.square(errors))))

        self.assertAlmostEqual(mae, 0.6, places=2)
        self.assertAlmostEqual(rmse, 0.7745966, places=4)

        accuracies = [max(0.0, 1.0 - err / max(1, act)) for err, act in zip(errors, actuals)]
        mean_acc = float(np.mean(accuracies)) * 100.0
        self.assertGreater(mean_acc, 85.0)

    def test_04_keyframe_evaluation_execution(self):
        """Verify evaluate_keyframes runs cleanly and outputs passing metrics."""
        benchmark = CCTVBenchmark(
            video_path=self.video_path,
            ground_truth_path=self.gt_path,
            roi_config_path=self.roi_path,
            conf_threshold=0.20,
            iou_threshold=0.50,
        )
        results = benchmark.evaluate_keyframes()

        self.assertNotIn("error", results)
        self.assertIn("evaluated_keyframes", results)
        self.assertIn("person_counting", results)
        self.assertIn("table_occupancy_classification", results)

        mae = results["person_counting"]["mean_absolute_error"]
        acc = results["person_counting"]["counting_accuracy_pct"]
        table_acc = results["table_occupancy_classification"]["overall_accuracy_pct"]

        # Benchmarks should meet realistic surveillance thresholds
        self.assertLess(mae, 1.5, f"MAE {mae} exceeded target threshold of 1.5")
        self.assertGreater(acc, 80.0, f"Person counting accuracy {acc}% below 80%")
        self.assertGreater(table_acc, 85.0, f"Table occupancy accuracy {table_acc}% below 85%")

    def test_05_markdown_report_generation(self):
        """Verify format_markdown_report generates well-formed markdown text."""
        dummy_results = {
            "timestamp": "2026-11-20T01:00:00Z",
            "benchmark_config": {
                "video": "sample.mp4",
                "model": "yolov8n.pt",
                "confidence_threshold": 0.20,
                "iou_threshold": 0.50,
                "image_size": 640,
                "device": "CPU",
            },
            "keyframe_accuracy": {
                "evaluated_keyframes": 10,
                "person_counting": {
                    "mean_absolute_error": 0.6,
                    "root_mean_squared_error": 0.8,
                    "counting_accuracy_pct": 90.0,
                },
                "table_occupancy_classification": {
                    "overall_accuracy_pct": 98.0,
                    "correct_evaluations": 49,
                    "total_evaluations": 50,
                    "per_table_accuracy_pct": {
                        "table_1": 100.0,
                        "table_2": 100.0,
                        "table_3": 90.0,
                        "table_4": 100.0,
                        "table_5": 100.0,
                    },
                },
            },
            "continuous_stream_profiling": {
                "timing_and_throughput": {
                    "overall_pipeline_fps": 26.5,
                    "mean_inference_time_ms": 35.2,
                    "median_inference_time_ms": 34.8,
                    "p95_inference_time_ms": 42.1,
                    "min_inference_time_ms": 28.3,
                    "max_inference_time_ms": 55.4,
                },
                "multi_object_tracking": {
                    "total_unique_tracks": 14,
                    "avg_persons_per_frame": 5.8,
                    "max_persons_in_single_frame": 7,
                    "avg_track_duration_sec": 18.5,
                    "max_track_duration_sec": 58.2,
                },
                "operations_and_flow": {
                    "final_queue_length": 0,
                    "avg_queue_wait_sec": 0.0,
                    "total_entries": 3,
                    "total_exits": 2,
                    "net_in_building": 1,
                },
                "final_table_occupancies": {
                    "table_1": {"state": "WAITING_FOR_FOOD", "occupants": 4, "tableware": False, "occupied_duration_sec": 45.0},
                },
            },
        }

        report = format_markdown_report(dummy_results)
        self.assertIn("# DineSight AI — Real CCTV Accuracy & Validation Benchmark Report", report)
        self.assertIn("Person Counting MAE", report)
        self.assertIn("Table State Classification", report)
        self.assertIn("Per-Table State Accuracy", report)


if __name__ == "__main__":
    unittest.main()
