"""
DineSight AI - Queue Monitoring Engine (Module 3C)
==================================================
Monitors patrons inside the designated entrance / host queue region.

Key Responsibilities:
- Identifies patrons entering the queue area using ByteTrack IDs.
- Enforces a minimum dwell-time filter so passers-by are not logged as queue customers.
- Tracks active waiting duration in real time for every individual in line.
- Implements a loss-grace period to handle momentary occlusion or detection drops.
- Records completed queue sessions upon exit and computes rolling average waiting time.
- Prevents duplicate counting of the same individual.
- Ephemeral memory only: No identities or facial records are retained.
"""

import time
from typing import Dict, List, Optional, Tuple


class QueuePersonSession:
    """
    Represents an individual customer's waiting session inside the queue zone.
    """

    def __init__(self, track_id: int, entry_time: float):
        self.track_id = int(track_id)
        self.entry_time = float(entry_time)
        self.last_seen_time = float(entry_time)
        self.exit_time: Optional[float] = None
        self.confirmed_waiting: bool = False
        self.is_active: bool = True

    def elapsed_wait(self, now: float) -> float:
        """Return elapsed waiting time in seconds."""
        if self.is_active:
            return max(0.0, now - self.entry_time)
        elif self.exit_time is not None:
            return max(0.0, self.exit_time - self.entry_time)
        return 0.0


class QueueMonitor:
    """
    Tracks real-time queue length, customer waiting times, and queue turnover.
    """

    def __init__(
        self,
        min_dwell_sec: float = 2.0,
        loss_grace_sec: float = 2.5,
        max_history: int = 100,
    ):
        """
        Args:
            min_dwell_sec: Minimum seconds a person must stay in queue ROI to be
                           counted as a confirmed waiting patron.
            loss_grace_sec: Grace period before marking an absent tracked person
                            as departed from the queue.
            max_history: Maximum completed sessions to retain in memory for statistics.
        """
        self.min_dwell_sec = float(min_dwell_sec)
        self.loss_grace_sec = float(loss_grace_sec)
        self.max_history = int(max_history)

        self.active_sessions: Dict[int, QueuePersonSession] = {}
        self.completed_sessions: List[QueuePersonSession] = []

    def reset(self) -> None:
        """Reset all queue history and active sessions."""
        self.active_sessions.clear()
        self.completed_sessions.clear()

    def update(
        self,
        queue_tracks: List[Dict],
        now: Optional[float] = None,
    ) -> Dict:
        """
        Update the queue state with the current frame's tracked occupants.

        Args:
            queue_tracks: List of track dicts inside the queue ROI.
            now: Current timestamp (seconds). Uses time.time() if None.

        Returns:
            Dictionary of real-time queue metrics:
            {
                "current_queue_length": 3,
                "confirmed_waiting_count": 2,
                "longest_current_wait_sec": 45.2,
                "avg_completed_wait_sec": 38.0,
                "total_completed_sessions": 5,
                "active_waiters": [
                    {"track_id": 1, "elapsed_sec": 45.2, "confirmed": True},
                    ...
                ]
            }
        """
        if now is None:
            now = time.time()

        current_track_ids = {trk["track_id"] for trk in queue_tracks if "track_id" in trk}

        # 1. Update existing sessions or register new entrants
        for tid in current_track_ids:
            if tid in self.active_sessions:
                sess = self.active_sessions[tid]
                sess.last_seen_time = now
                if not sess.confirmed_waiting:
                    if (now - sess.entry_time) >= self.min_dwell_sec:
                        sess.confirmed_waiting = True
            else:
                # New person arrived in the queue area
                new_sess = QueuePersonSession(track_id=tid, entry_time=now)
                # If dwell is 0, immediately confirmed
                if self.min_dwell_sec <= 0:
                    new_sess.confirmed_waiting = True
                self.active_sessions[tid] = new_sess

        # 2. Check for departures (patrons not seen in current frame)
        to_remove = []
        for tid, sess in self.active_sessions.items():
            if tid not in current_track_ids:
                if (now - sess.last_seen_time) >= self.loss_grace_sec:
                    # Patron departed the queue
                    sess.exit_time = sess.last_seen_time
                    sess.is_active = False
                    # Only log completed session if they were confirmed waiting
                    duration = sess.exit_time - sess.entry_time
                    if sess.confirmed_waiting and duration >= self.min_dwell_sec:
                        self.completed_sessions.append(sess)
                        if len(self.completed_sessions) > self.max_history:
                            self.completed_sessions.pop(0)
                    to_remove.append(tid)

        for tid in to_remove:
            del self.active_sessions[tid]

        # 3. Calculate Real-Time Metrics
        active_waiters = []
        confirmed_count = 0
        longest_current_wait = 0.0

        for tid, sess in self.active_sessions.items():
            elapsed = round(sess.elapsed_wait(now), 1)
            is_conf = sess.confirmed_waiting or (now - sess.entry_time >= self.min_dwell_sec)
            if is_conf:
                confirmed_count += 1
                if elapsed > longest_current_wait:
                    longest_current_wait = elapsed

            active_waiters.append({
                "track_id": tid,
                "elapsed_sec": elapsed,
                "confirmed": is_conf,
            })

        # Average wait of completed sessions
        avg_completed_wait = 0.0
        if self.completed_sessions:
            total_time = sum(s.elapsed_wait(s.exit_time or now) for s in self.completed_sessions)
            avg_completed_wait = round(total_time / len(self.completed_sessions), 1)

        return {
            "current_queue_length": len(self.active_sessions),
            "confirmed_waiting_count": confirmed_count,
            "longest_current_wait_sec": round(longest_current_wait, 1),
            "avg_completed_wait_sec": avg_completed_wait,
            "total_completed_sessions": len(self.completed_sessions),
            "active_waiters": sorted(active_waiters, key=lambda x: x["elapsed_sec"], reverse=True),
        }

