"""
DineSight AI - Synthetic Restaurant Video Generator
===================================================
Generates a realistic 15-second 1280x720 30-FPS MP4 test video representing
a restaurant layout with:
- Restaurant entrance / queue zone.
- Three numbered dining tables with chairs.
- Animated simulated patrons walking from the entrance to tables and waiting in queue.
- Timestamp clock.

This enables immediate testing of the video ingestion, frame sampling,
and downstream ROI / detection pipelines without requiring external video downloads.
"""

import os
import cv2
import numpy as np


def generate_restaurant_video(
    output_path: str = "data/sample_videos/restaurant_sample.mp4",
    duration_sec: int = 15,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> str:
    """
    Generate and save a synthetic restaurant surveillance video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path: {output_path}")

    total_frames = duration_sec * fps
    print(f"[Generator] Generating {total_frames} frames ({duration_sec}s @ {fps}fps) to '{output_path}'...")

    # Table layout definitions
    tables = [
        {"id": "Table 1", "pos": (450, 240), "radius": 70, "color": (140, 110, 80)},
        {"id": "Table 2", "pos": (850, 240), "radius": 70, "color": (140, 110, 80)},
        {"id": "Table 3", "pos": (650, 480), "radius": 85, "color": (130, 100, 75)},
    ]

    for frame_idx in range(total_frames):
        # 1. Background floor (warm wood tone)
        frame = np.full((height, width, 3), (35, 42, 50), dtype=np.uint8)

        # Floor tile grid pattern
        for y in range(0, height, 80):
            cv2.line(frame, (0, y), (width, y), (42, 50, 60), 1)
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (42, 50, 60), 1)

        # 2. Entrance / Host Queue Area (left corridor)
        queue_rect = (50, 120, 220, 520)  # x1, y1, x2, y2
        cv2.rectangle(frame, (queue_rect[0], queue_rect[1]), (queue_rect[2], queue_rect[3]), (55, 65, 75), -1)
        cv2.rectangle(frame, (queue_rect[0], queue_rect[1]), (queue_rect[2], queue_rect[3]), (90, 120, 140), 2)
        cv2.putText(
            frame,
            "ENTRANCE / QUEUE",
            (queue_rect[0] + 15, queue_rect[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 200, 210),
            1,
            cv2.LINE_AA,
        )

        # 3. Draw Dining Tables and Surrounding Chairs
        for t in tables:
            tx, ty = t["pos"]
            tr = t["radius"]

            # Chairs around table
            chair_offsets = [(0, -tr - 25), (0, tr + 25), (-tr - 25, 0), (tr + 25, 0)]
            for cx_off, cy_off in chair_offsets:
                cv2.circle(frame, (tx + cx_off, ty + cy_off), 18, (70, 75, 85), -1)
                cv2.circle(frame, (tx + cx_off, ty + cy_off), 18, (110, 115, 125), 2)

            # Table top
            cv2.circle(frame, (tx, ty), tr, t["color"], -1)
            cv2.circle(frame, (tx, ty), tr, (170, 140, 100), 3)

            # Table label
            cv2.putText(
                frame,
                t["id"],
                (tx - 35, ty + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

        # 4. Animated Simulated People
        # Person 1 (Waiting in queue)
        p1_x = 130
        p1_y = 250 + int(np.sin(frame_idx * 0.05) * 8)
        cv2.circle(frame, (p1_x, p1_y), 22, (200, 100, 50), -1)  # Head / body
        cv2.putText(frame, "P1", (p1_x - 10, p1_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Person 2 (Standing in queue behind P1)
        p2_x = 130
        p2_y = 380 + int(np.cos(frame_idx * 0.04) * 5)
        cv2.circle(frame, (p2_x, p2_y), 22, (180, 80, 60), -1)
        cv2.putText(frame, "P2", (p2_x - 10, p2_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Person 3 (Walking from queue to Table 1, then sitting down)
        walk_progress = min(1.0, frame_idx / (total_frames * 0.5))
        p3_x = int(140 + walk_progress * (450 - 140))
        p3_y = int(480 + walk_progress * (240 - 480))
        cv2.circle(frame, (p3_x, p3_y), 24, (50, 150, 220), -1)
        cv2.putText(frame, "P3", (p3_x - 10, p3_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Person 4 (Already seated at Table 2 dining)
        p4_x = 850
        p4_y = 240 - 70 - 15
        cv2.circle(frame, (p4_x, p4_y), 22, (80, 180, 80), -1)
        cv2.putText(frame, "P4", (p4_x - 10, p4_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # Cups on Table 2
        cv2.circle(frame, (830, 230), 8, (230, 230, 230), -1)
        cv2.circle(frame, (865, 245), 10, (200, 200, 200), -1)

        # 5. Timestamp Watermark (CCTV Style)
        time_sec = frame_idx / fps
        cctv_time = f"CAM-01 [DINING-HALL-A]  2026-11-05 13:45:{int(time_sec):02d}"
        cv2.putText(
            frame,
            cctv_time,
            (width - 520, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()
    print(f"[Generator] Successfully created: '{output_path}' ({os.path.getsize(output_path)} bytes)")
    return output_path


if __name__ == "__main__":
    generate_restaurant_video()

