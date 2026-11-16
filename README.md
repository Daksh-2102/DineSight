# DineSight AI
## AI-Powered Computer Vision & Video Analytics for Intelligent Restaurant Operations

DineSight AI transforms standard restaurant CCTV video into an intelligent operational monitoring platform.

---

## Project Status

- [x] **Phase 1: Environment Setup & Video Ingestion** (OpenCV, Frame Resizing, Sampling, FPS HUD)
- [x] **Phase 2: YOLOv8 Object Detection & ByteTrack Tracking** (Person Detection, Persistent IDs, Telemetry)
- [x] **Phase 3: Spatial ROI Mapping, Table-State Engine & Queue Monitoring** (Polygons, Dwell-Time, Vacant/Occupied/Dining, Queue Waiting Times)
- [ ] **Phase 4: SQLite Database & FastAPI Backend** (Event Persistence, REST & WebSocket Streaming API)
- [ ] **Phase 5: React + Vite Web Dashboard** (Live CCTV, Floor Plan, Recharts Analytics)
- [ ] **Phase 6: Testing, Evaluation & Presentation Documentation**

---

## Quick Start (Master Integrated Application)

Run the unified, presentation-ready application containing all completed phases:

```powershell
# Run the complete integrated system (Phase 1 + Phase 2 + Phase 3)
.\.venv\Scripts\python.exe run.py

# Live Mode Switcher:
# Press 'm' while the window is open to cycle: RAW -> DETECT -> TRACK -> FULL!

# Or launch directly into a specific mode:
.\.venv\Scripts\python.exe run.py --mode full     # Full System: Tables + Queue + Tracking
.\.venv\Scripts\python.exe run.py --mode track    # Phase 2: ByteTrack Tracking
.\.venv\Scripts\python.exe run.py --mode detect   # Phase 2: YOLOv8 Detections Only
.\.venv\Scripts\python.exe run.py --mode raw      # Phase 1: Video Ingestion Baseline
```

---

## Phase 3: Spatial ROI, Table-State Engine & Queue Monitoring

### Key Features
1. **Configurable Spatial ROIs (`config/rois.json`)**:
   - Arbitrary polygon definitions for individual dining tables, entrance vestibule, and host queue area.
   - Normalized coordinates (`[0.0, 1.0]`) ensure spatial boundaries scale dynamically across any video resolution.
2. **Robust Point-in-Polygon Association**:
   - Uses `cv2.pointPolygonTest` (Ray Casting) with bottom-center projection (`(bottom_cx, bottom_cy)`), aligning with 2.5D CCTV camera perspective where patrons touch the floor/seating plane.
3. **Table Finite State Machine & Temporal Smoothing**:
   - `VACANT`: Green polygon fill.
   - `OCCUPIED`: Red polygon fill (confirmed after `dwell_time_sec`).
   - `WAITING_FOR_FOOD`: Orange polygon fill (seated patrons, no tableware detected).
   - `DINING`: Royal blue polygon fill (seated patrons + tableware evidence).
   - **Dwell Time Filter**: Passers-by walking past a table for $< 2.0\text{s}$ do not falsely occupy it.
   - **Release Grace Period**: Momentary occlusion or tracking blips for $< 2.5\text{s}$ do not prematurely reset a table to vacant.
4. **Entrance Queue Monitoring**:
   - Live queue count, elapsed waiting times for active patrons, and average wait time for completed visits.
   - Dwell-time filtering prevents entrance passer-by miscounts.

---

### Project Structure (Phase 3)

```text
DineSight/
├── .env                                  # Configuration (resolutions, default video path)
├── .env.example                          # Environment template
├── .gitignore                            # Excludes venv, heavy cache, models
├── README.md                             # Project setup and usage documentation
├── run.py                                # Master Application (All Phases Merged)
├── run_phase1.py                         # Phase 1 launcher (Video Ingestion)
├── run_phase2.py                         # Phase 2 launcher (YOLOv8 + ByteTrack)
├── run_phase3.py                         # Phase 3 launcher (ROIs + Table States + Queue)
│
├── config/
│   └── rois.json                         # Spatial ROI polygon configurations
│
├── backend/
│   ├── requirements.txt                  # Dependencies
│   └── app/
│       ├── __init__.py
│       └── core/
│           ├── __init__.py               # Core module exports
│           ├── video_reader.py           # Module 1: VideoReader & HUD
│           ├── detector.py               # Module 2: YOLODetector (YOLOv8)
│           ├── tracker.py                # Module 2: ByteTracker (ByteTrack)
│           ├── roi_manager.py            # Module 3A: ROIManager & RegionOfInterest
│           ├── state_engine.py           # Module 3B: TableStateEngine & FSM
│           ├── queue_monitor.py          # Module 3C: QueueMonitor & Wait Sessions
│           └── generate_sample_video.py  # Test video generator
│
├── data/
│   ├── models/
│   │   └── yolov8n.pt                    # Pretrained YOLOv8 weights (6.2 MB)
│   └── sample_videos/
│       ├── pedestrians_surveillance.mp4  # Real surveillance test footage
│       └── restaurant_sample.mp4         # Synthetic restaurant layout
│
└── tests/
    ├── test_video_reader.py              # Phase 1 test suite (6/6 passing)
    ├── test_detector.py                  # Phase 2 detector tests (7/7 passing)
    ├── test_tracker.py                   # Phase 2 tracker tests (6/6 passing)
    ├── test_roi_manager.py               # Phase 3A ROI tests (6/6 passing)
    ├── test_state_engine.py              # Phase 3B Table State tests (6/6 passing)
    ├── test_queue_monitor.py             # Phase 3C Queue Monitor tests (5/5 passing)
    └── test_phase3_integration.py        # Phase 3 End-to-End integration tests (1/1 passing)
```

---

### How to Run Phase 3

```powershell
# 1. Run Phase 3 Dedicated Interactive Stream
.\.venv\Scripts\python.exe run_phase3.py

# 2. Run with Custom Dwell Time (e.g. 3.0s dwell, 3.5s release)
.\.venv\Scripts\python.exe run_phase3.py --dwell 3.0 --release 3.5

# 3. Run on Live Webcam
.\.venv\Scripts\python.exe run_phase3.py --source 0

# 4. Run the Full Test Suite (37 tests across all modules)
.\.venv\Scripts\python.exe tests/test_video_reader.py
.\.venv\Scripts\python.exe tests/test_detector.py
.\.venv\Scripts\python.exe tests/test_tracker.py
.\.venv\Scripts\python.exe tests/test_roi_manager.py
.\.venv\Scripts\python.exe tests/test_state_engine.py
.\.venv\Scripts\python.exe tests/test_queue_monitor.py
.\.venv\Scripts\python.exe tests/test_phase3_integration.py
```
