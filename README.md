# DineSight AI
## AI-Powered Computer Vision & Video Analytics for Intelligent Restaurant Operations

DineSight AI transforms standard restaurant CCTV video into an intelligent operational monitoring platform.

---

## Phase 1: Video Ingestion & Telemetry Pipeline

### Overview
Phase 1 implements the foundational video streaming layer using OpenCV. It provides:
- Support for recorded video files (`.mp4`, `.avi`), live webcams, and RTSP streams.
- Configurable frame resizing to balance quality and inference speed.
- Dynamic frame sampling (skip $N$ frames to conserve processing overhead).
- Real-time smoothed FPS (Frames Per Second) tracking via Exponential Moving Average.
- Semi-transparent Heads-Up Display (HUD) overlay with stream telemetry.
- Robust error handling for missing files, stream disconnects, and EOF management.

---

### Project Structure (Phase 1)

```text
DineSight/
├── .env                        # Active environment configuration
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Project dependencies
├── run_phase1.py               # Launcher script for Phase 1 preview
│
├── backend/
│   ├── requirements.txt        # Backend dependencies
│   └── app/
│       ├── __init__.py
│       └── core/
│           ├── __init__.py
│           ├── video_reader.py         # VideoReader engine & HUD
│           └── generate_sample_video.py# Synthetic test video generator
│
├── data/
│   ├── sample_videos/
│   │   └── restaurant_sample.mp4       # Generated test restaurant video
│   └── models/                         # Reserved for YOLOv8 weights
│
└── tests/
    └── test_video_reader.py    # Automated test suite (6 passing unit tests)
```

---

### How to Run Phase 1

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Run the Video Ingestion Preview**:
   ```powershell
   python run_phase1.py
   ```

3. **Interactive Keyboard Controls**:
   - `[SPACE]` or `[p]`: Pause / Resume playback
   - `[r]`: Rewind / Restart video from frame 0
   - `[+]` / `[-]`: Increase / Decrease frame skip rate dynamically
   - `[q]` or `[ESC]`: Exit preview window

4. **Run with Custom Options**:
   ```powershell
   # Use a custom video file
   python run_phase1.py --source "path/to/video.mp4"

   # Use your laptop webcam (device 0)
   python run_phase1.py --source 0

   # Process every 2nd frame at 960x540
   python run_phase1.py --skip 2 --width 960 --height 540
   ```

5. **Run the Automated Unit Tests**:
   ```powershell
   python tests/test_video_reader.py
   ```

