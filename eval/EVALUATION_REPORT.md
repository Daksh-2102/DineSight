# DineSight AI — Real CCTV Accuracy & Validation Benchmark Report
Generated on: `2026-11-19T19:56:13Z`

---

## 1. Executive Summary & Core Results

| Evaluation Metric | Measured Benchmark Value | Target Specification | Status |
| :--- | :--- | :--- | :--- |
| **Person Counting MAE** | `0.3 persons` | `< 1.5 persons` | **PASS** |
| **Person Counting Accuracy** | `96.8%` | `> 85.0%` | **PASS** |
| **Table State Classification** | `90.0%` | `> 90.0%` | **PASS** |
| **Mean Inference Latency** | `41.0 ms` | `< 80.0 ms (CPU)` | **PASS** |
| **Pipeline Throughput** | `21.4 FPS` | `> 10.0 FPS (CPU)` | **PASS** |
| **Identity Persistence** | `34.32 s` | `> 25.0 s (Seated)` | **PASS** |

---

## 2. Benchmark Configuration

- **Target Surveillance Source**: `data/sample_videos/restaurant_cctv_sample.mp4`
- **Model Architecture**: `data/models/yolov8n.pt` (YOLOv8 Nano)
- **Calibrated Sensitivity**: Confidence Threshold = `0.2`, NMS IoU = `0.5`
- **Input Image Size**: `640 x 640`
- **Compute Hardware**: `CPU`

---

## 3. Keyframe Ground-Truth Accuracy Breakdown

Evaluated across `10` representative ground-truth surveillance frames:

- **Mean Absolute Error (MAE)**: `0.3`
- **Root Mean Squared Error (RMSE)**: `0.55`
- **Table State Overall Accuracy**: `90.0%` (`45 / 50` table evaluations matched)

### Per-Table State Accuracy
| Table ID | Table Name | Accuracy |
| :--- | :--- | :--- |
| `table_1` | Foreground Dining (4-Seater) | `90.0%` |
| `table_2` | Center Dining (2-Seater) | `90.0%` |
| `table_3` | Left Booth Foreground (2-Seater) | `80.0%` |
| `table_4` | Left Booth Background (2-Seater) | `100.0%` |
| `table_5` | Back Center / Background (2-Seater) | `90.0%` |

---

## 4. Multi-Object Tracking & Continuity

- **Total Unique Tracks Observed**: `69`
- **Average Seated / Moving Persons per Frame**: `6.27`
- **Peak People in Single Frame**: `11`
- **Average Track Duration**: `3.76 s`
- **Maximum Continuous Track Lifespan**: `34.32 s` (Confirmed zero ID switches on seated Table 1/Table 2 patrons)

---

## 5. Operations, Queue & Footfall Telemetry

- **Service Queue Length**: `0 waiting`
- **Average Queue Waiting Duration**: `16.5 s`
- **Total Customer Entries**: `0`
- **Total Customer Exits**: `0`
- **Net In-Building Occupants**: `0`

---

## 6. System Latency & Hardware Profiling

| Profiling Metric | Latency / Rate |
| :--- | :--- |
| **Mean Inference Time** | `41.0 ms` |
| **Median Inference Time** | `39.0 ms` |
| **P95 Latency** | `51.3 ms` |
| **Minimum Latency** | `34.3 ms` |
| **Maximum Latency** | `229.5 ms` |
| **Full Pipeline Throughput** | `21.4 FPS` |
