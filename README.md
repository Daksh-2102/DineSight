# DineSight AI
## AI-Powered Computer Vision & Video Analytics for Intelligent Restaurant Operations

DineSight AI transforms standard restaurant CCTV video into an intelligent operational monitoring platform.

---

## Project Status

- [x] **Phase 1: Environment Setup & Video Ingestion** (OpenCV, Frame Resizing, Sampling, FPS HUD)
- [x] **Phase 2: YOLOv8 Object Detection & ByteTrack Tracking** (Person Detection, Persistent IDs, Telemetry)
- [ ] **Phase 3: Spatial ROI Mapping & Table-State Recognition** (Vacant, Occupied, Waiting, Dining)
- [ ] **Phase 4: SQLite Database & FastAPI Backend** (Event Logging, Analytics REST API)
- [ ] **Phase 5: React + Vite Web Dashboard** (Floor Plan, Live CCTV, Recharts Trends)
- [ ] **Phase 6: Testing, Evaluation & Presentation Documentation**

---

## Quick Start (Unified Master Application)

To run the unified application that combines both phases:

```powershell
# Run the complete integrated tracking pipeline (Default)
.\.venv\Scripts\python.exe run.py

# Switch between modes dynamically:
# Press 'm' while the window is open to cycle: RAW -> DETECT -> TRACK!

# Or launch directly into a specific mode:
.\.venv\Scripts\python.exe run.py --mode raw      # Phase 1: Video Ingestion Only
.\.venv\Scripts\python.exe run.py --mode detect   # Phase 2: YOLOv8 Detections Only
.\.venv\Scripts\python.exe run.py --mode track    # Phase 2: Full ByteTrack Tracking
```

---

## Phase 2: Object Detection & Multi-Object Tracking

### Overview
Phase 2 implements the visual intelligence and temporal memory layers of DineSight AI:
- **YOLOv8 Nano (`yolov8n.pt`)**: Real-time object detection focused initially on the `person` class.
- **ByteTrack Tracker**: Associates detections across consecutive frames to maintain temporary integer tracking IDs (`Person #1`, `Person #2`), eliminating duplicate counting.
- **Telemetry HUD**: Displays compute device (CPU vs CUDA GPU), pure model inference latency, end-to-end pipeline FPS, active detected people count, and total unique individuals tracked.
- **Privacy-by-Design**: Tracking IDs are temporary in-memory tokens. No facial recognition or permanent biometric data is stored.

---

### Project Structure (Phase 2)

```text
DineSight/
├── .env                                  # Configuration (resolutions, default video path)
├── .env.example                          # Environment template
├── .gitignore                            # Excludes venv, heavy cache, models
├── README.md                             # Project setup and usage documentation
├── run_phase1.py                         # Phase 1 launcher (Video Ingestion)
├── run_phase2.py                         # Phase 2 launcher (YOLOv8 + ByteTrack)
│
├── backend/
│   ├── requirements.txt                  # Backend dependencies (OpenCV, YOLO, ByteTrack)
│   └── app/
│       ├── __init__.py
│       └── core/
│           ├── __init__.py
│           ├── video_reader.py           # Module 1: VideoReader & HUD
│           ├── detector.py               # Module 2: YOLODetector wrapper
│           ├── tracker.py                # Module 2: ByteTracker wrapper
│           └── generate_sample_video.py  # Test video generator
│
├── data/
│   ├── models/
│   │   └── yolov8n.pt                    # Pretrained YOLOv8 weights (6.2 MB)
│   └── sample_videos/
│       ├── pedestrians_surveillance.mp4  # Surveillance test video with real people
│       └── restaurant_sample.mp4         # Synthetic layout sample
│
└── tests/
    ├── test_video_reader.py              # Phase 1 test suite (6/6 passing)
    ├── test_detector.py                  # Phase 2 detector tests (7/7 passing)
    └── test_tracker.py                   # Phase 2 tracker tests (6/6 passing)
```

---

### How to Run Phase 2

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Run YOLOv8 + ByteTrack Interactive Live Stream**:
   ```powershell
   python run_phase2.py
   ```

3. **Interactive Keyboard Controls**:
   - `[SPACE]` or `[p]`: Pause / Resume tracking
   - `[r]`: Rewind video and reset tracker state
   - `[+]` / `[-]`: Increase / Decrease frame skip dynamically
   - `[q]` or `[ESC]`: Exit window

4. **Custom Options**:
   ```powershell
   # Run with a webcam
   python run_phase2.py --source 0

   # Custom confidence threshold & frame sampling
   python run_phase2.py --conf 0.40 --skip 2

   # Force CPU mode
   python run_phase2.py --device cpu
   ```

5. **Run the Automated Test Suite**:
   ```powershell
   # Run all Phase 1 & Phase 2 tests
   python tests/test_video_reader.py
   python tests/test_detector.py
   python tests/test_tracker.py
   ```
