# DineSight AI
## Computer Vision & Video Analytics for Intelligent Restaurant Operations

DineSight AI transforms standard restaurant CCTV video streams into an automated, real-time operational intelligence platform. Built with PyTorch, Ultralytics YOLOv8, ByteTrack, and OpenCV, DineSight continuously monitors dining room table occupancy, service queues, customer wait durations, entrance footfall rates, and table-turnover activity without requiring intrusive hardware sensors.

---

## Project Status

- [x] **Phase 1: Video Ingestion & Telemetry** (OpenCV Ingestion, Aspect-Ratio Scaling, Frame Sampling, FPS Calculation)
- [x] **Phase 2: Object Detection & Multi-Object Tracking** (YOLOv8 Nano, ByteTrack 2-Stage Association, ID Persistence)
- [x] **Phase 3: Spatial ROI Mapping, Table-State Engine, Queue & Footfall** (Perspective ROIs, Hybrid Spatial Association, Grounded Table FSM, Directional Entrance Tripwire, Tableware Detection)
- [x] **Surveillance Calibration & Empirical Validation** (12 Forensic Engineering Milestones, 96.8% Counting Accuracy, 90.0% Table Accuracy, 21.4 FPS CPU Throughput)
- [ ] **Phase 4: SQLite Database & FastAPI Backend** (Event Persistence, REST Endpoints & WebSocket Event Streaming)
- [ ] **Phase 5: React + Vite Web Dashboard** (Live CCTV Video Stream, Interactive Floor Plan, Real-time Operations HUD, Historical Recharts Analytics)
- [ ] **Phase 6: Final Verification, Presentation & Deployment Documentation**

---

## System Architecture & Processing Pipeline

```
           [ Restaurant CCTV Stream / RTSP / MP4 / Webcam ]
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Module 1: VideoReader │ (OpenCV Ingestion & Frame Sampling)
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Module 2: YOLODetector │ (YOLOv8 Single-Pass Multi-Class Inference)
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Module 2: ByteTracker  │ (2-Stage Association, 60-Frame Buffer)
                     └────────────┬────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  ROIManager      │    │  QueueMonitor    │    │  EntranceMonitor │
│  - Hybrid Spatial│    │  - Service Line  │    │  - Directional   │
│    Association   │    │  - Order Counter │    │    Tripwire      │
│  - Tableware Map │    │  - Wait Timers   │    │  - Flow Rate/Min │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                        │                        │
         ▼                        │                        │
┌──────────────────┐              │                        │
│ TableStateEngine │              │                        │
│ - Dwell Time     │              │                        │
│ - Release Grace  │              │                        │
│ - Activity FSM   │              │                        │
└────────┬─────────┘              │                        │
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Master Telemetry HUD    │ (Dark Glassmorphism 3-Column Overlay)
                     └─────────────────────────┘
```

---

## Quick Start (Master Application Runner)

Run the unified, production-grade master runner containing all operational modes:

```powershell
# Activate Python Virtual Environment
.\.venv\Scripts\Activate.ps1

# Launch Complete System (Full Analytics: Tables + Queue + Entrance + HUD)
.\.venv\Scripts\python.exe run.py

# Live Mode Switcher:
# Press 'm' at runtime to dynamically cycle: RAW -> DETECT -> TRACK -> FULL!

# Or launch directly into a specific mode:
.\.venv\Scripts\python.exe run.py --mode full     # Full System: Tables + Queue + Tracking + Footfall
.\.venv\Scripts\python.exe run.py --mode track    # Phase 2: ByteTrack Multi-Object Tracking
.\.venv\Scripts\python.exe run.py --mode detect   # Phase 2: YOLOv8 Bounding Box Detections
.\.venv\Scripts\python.exe run.py --mode raw      # Phase 1: Video Ingestion Baseline
```

### Master Runner Keyboard Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`m`** | **Cycle Modes** | Live toggle between `RAW`, `DETECT`, `TRACK`, and `FULL` modes without restarting. |
| **`SPACE` / `p`** | **Pause / Resume** | Freeze playback to inspect bounding boxes and table state assignments. |
| **`r`** | **Rewind & Reset** | Rewind video to frame 0 and atomically reset all state machines, queues, and track IDs. |
| **`+` / `-`** | **Sampling Step** | Dynamically adjust frame skip rate ($1/1, 1/2, 1/3$) to optimize CPU load. |
| **`q` / `ESC`** | **Quit** | Gracefully release video capture devices and close display windows. |

---

## Forensic Calibration & Validation Results

Across 12 rigorous engineering milestones, DineSight AI was calibrated and evaluated on authentic 848x480 surveillance footage (`data/sample_videos/restaurant_cctv_sample.mp4`).

### Quantitative Evaluation vs. Engineering Specifications

| Evaluation Metric | Measured Benchmark Value | Project Target Specification | Validation Status |
| :--- | :--- | :--- | :---: |
| **Person Counting MAE** | **`0.3 persons`** | $< 1.5\text{ persons}$ | **PASS** |
| **Person Counting Accuracy** | **`96.8%`** | $> 85.0\%$ | **PASS** |
| **Table State Classification** | **`90.0%`** ($45/50$ table matches) | $\ge 90.0\%$ | **PASS** |
| **Mean Inference Latency** | **`41.0 ms`** | $< 80.0\text{ ms (CPU)}$ | **PASS** |
| **Full Pipeline Throughput** | **`21.4 FPS`** | $> 10.0\text{ FPS (CPU)}$ | **PASS** |
| **Identity Persistence** | **`34.32 s`** continuous track | $> 25.0\text{ s (Seated)}$ | **PASS** |

### Key Technological Achievements
1. **Hybrid Person-to-Table Association**: Combines rasterized bounding box polygon intersection area ($\ge 25\%$ overlap) with bottom-center floor projection and multi-anchor fallback points. Eliminates misclassification of patrons leaning or sitting across boundary edges.
2. **Grounded Finite State Machine**: Decouples primary ground-truth occupancy (`VACANT`, `OCCUPIED`) from secondary activity heuristics (`ESTIMATED_WAITING`, `ESTIMATED_DINING`). Requires $2.0\text{s}$ dwell before occupancy confirmation and $3.0\text{s}$ release grace period to prevent flickering during momentary occlusion.
3. **Zero-Latency Single-Pass Multi-Class Inference**: Detects patrons (`person`) and dining utensils (`bottle`, `cup`, `fork`, `knife`, `spoon`, `bowl`) concurrently in YOLOv8 with zero latency penalty ($41.0\text{ ms}$ on CPU).
4. **Directional Entrance Tripwire**: Spatial hysteresis buffer ($\pm 15\text{ px}$) and $3.0\text{s}$ debounce prevent doorway loiterers from corrupting customer footfall counts.

---

## Interactive Utilities & Evaluation Tools

### 1. Interactive ROI Calibration Utility
Calibrate spatial table coordinates and perspective polygons using an interactive GUI or automated non-interactive presets:

```powershell
# Launch interactive GUI calibrator
.\.venv\Scripts\python.exe tools/calibrate_rois.py

# Apply perspective-calibrated CCTV preset non-interactively
.\.venv\Scripts\python.exe tools/calibrate_rois.py --preset cctv_calibrated
```

### 2. CCTV Accuracy Benchmark Harness
Run the automated quantitative evaluation harness across ground-truth surveillance keyframes and continuous video streams:

```powershell
# Run keyframe accuracy and continuous stream profiling
.\.venv\Scripts\python.exe eval/evaluate_accuracy.py --sample-step 2 --max-frames 800

# Results are automatically serialized to:
# - eval/evaluation_results.json
# - eval/EVALUATION_REPORT.md
```

---

## Project Structure

```text
DineSight/
├── .env                                  # Configuration variables (video source, resolution)
├── .env.example                          # Environment template
├── .gitignore                            # Excludes venv, heavy cache, models
├── README.md                             # Comprehensive technical documentation
├── run.py                                # Master Application Runner (All Modes Merged)
├── run_phase1.py                         # Phase 1 standalone launcher (Video Ingestion)
├── run_phase2.py                         # Phase 2 standalone launcher (Detection & Tracking)
├── run_phase3.py                         # Phase 3 standalone launcher (Table States & Queue)
│
├── config/
│   ├── rois.json                         # Perspective-calibrated polygon definitions
│   └── bytetrack_dinesight.yaml          # Calibrated ByteTrack parameters (60-frame buffer)
│
├── eval/
│   ├── evaluate_accuracy.py              # Automated accuracy benchmark runner
│   ├── ground_truth_keyframes.json       # Ground-truth keyframe annotations
│   ├── evaluation_results.json           # Serialized benchmark metrics
│   └── EVALUATION_REPORT.md              # Markdown accuracy report
│
├── tools/
│   └── calibrate_rois.py                 # Interactive ROI calibration utility
│
├── backend/
│   ├── requirements.txt                  # Python dependencies
│   └── app/
│       ├── __init__.py
│       └── core/
│           ├── __init__.py               # Core package exports
│           ├── video_reader.py           # Module 1: Video ingestion & HUD
│           ├── detector.py               # Module 2: YOLOv8 person & tableware detector
│           ├── tracker.py                # Module 2: ByteTrack persistent multi-object tracker
│           ├── roi_manager.py            # Module 3A: Spatial ROI polygon manager
│           ├── state_engine.py           # Module 3B: Table-state finite state machine
│           ├── queue_monitor.py          # Module 3C: Service counter & queue wait-time tracker
│           ├── entrance_monitor.py       # Module 3D: Directional footfall tripwire monitor
│           └── generate_sample_video.py  # Synthetic sample video generator
│
├── data/
│   ├── models/
│   │   └── yolov8n.pt                    # Pretrained YOLOv8 weights (6.2 MB)
│   └── sample_videos/
│       ├── restaurant_cctv_sample.mp4    # Real CCTV restaurant surveillance footage
│       ├── pedestrians_surveillance.mp4  # Surveillance test footage
│       └── restaurant_sample.mp4         # Synthetic layout sample
│
└── tests/
    ├── test_video_reader.py              # VideoReader unit tests (6 tests)
    ├── test_detector.py                  # YOLODetector unit tests (12 tests)
    ├── test_tracker.py                   # ByteTracker unit tests (9 tests)
    ├── test_roi_manager.py               # ROIManager unit tests (10 tests)
    ├── test_state_engine.py              # TableStateEngine unit tests (9 tests)
    ├── test_queue_monitor.py             # QueueMonitor unit tests (7 tests)
    ├── test_entrance_monitor.py          # EntranceMonitor unit tests (10 tests)
    ├── test_phase3_integration.py        # End-to-end integration tests (2 tests)
    ├── test_evaluation.py                # Benchmark & ground-truth tests (5 tests)
    └── test_master_runner.py             # Master runner & CLI integration tests (7 tests)
```

---

## Automated Test Suite (77 / 77 Tests Passing)

Execute the complete project test suite covering all modules, state machines, tracking algorithms, and runners:

```powershell
# Run the entire test suite via unittest discovery
.\.venv\Scripts\python.exe -m unittest discover tests

# Output:
# Ran 77 tests in ~50s
# OK (All 77 tests passing, 0 failures, 0 errors)
```

---

## Next Steps: Phase 4 Implementation Plan

With Phase 1, Phase 2, and Phase 3 completed, audited, calibrated, and validated with empirical benchmarks, the system is prepared for Phase 4:

1. **SQLite Database Schema**:
   - `tables`: Static metadata (table ID, name, seat capacity, coordinate bounds).
   - `table_events`: State transitions (`VACANT` $\leftrightarrow$ `OCCUPIED` $\leftrightarrow$ `DINING`), occupant count, duration, timestamp.
   - `queue_events`: Waiting session records (arrival, service start, departure, wait duration).
   - `footfall_events`: Directional entries/exits with timestamps.
2. **FastAPI Application & REST Endpoints**:
   - `GET /api/tables`: Current occupancy statuses, occupant counts, elapsed durations.
   - `GET /api/queue`: Real-time queue length, active wait times, average service time.
   - `GET /api/analytics/summary`: Daily footfall, average table turn time, peak occupancy hours.
3. **WebSocket Event Streaming**:
   - `WS /ws/live`: Real-time telemetry broadcast push to client web dashboards at 10-25 Hz.
