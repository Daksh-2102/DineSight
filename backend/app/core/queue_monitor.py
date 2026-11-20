"""
DineSight AI - Queue & Service Counter Monitoring Engine (Module 3C)
====================================================================
Monitors patrons inside the designated service counter and waiting queue zones.

Key Responsibilities:
- Distinguishes between customers actively ordering at the service counter vs.
  customers waiting in line in the queue.
- Enforces a minimum dwell-time filter so transient passers-by are not logged as queue patrons.
- Tracks active waiting duration in real time for every individual in line.
- Tracks transaction service duration for customers at the counter.
- Implements a loss-grace period to handle momentary occlusion or detection drops.
- Records completed queue and service sessions to compute rolling averages.
- Ephemeral in-memory telemetry: No identities, facial profiles, or biometric records.
"""

import time
from typing import Dict, List, Optional, Tuple


class QueuePersonSession:
    """
    Represents an individual customer's waiting session inside the queue or counter zone.
    """

    def __init__(self, track_id: int, entry_time: float, zone: str = "queue"):
        self.track_id = int(track_id)
        self.entry_time = float(entry_time)
        self.last_seen_time = float(entry_time)
        self.exit_time: Optional[float] = None
        self.confirmed_waiting: bool = False
        self.is_active: bool = True
        self.zone: str = str(zone)

    def elapsed_wait(self, now: float) -> float:
        """Return elapsed waiting or service time in seconds."""
        if self.is_active:
            return max(0.0, now - self.entry_time)
        elif self.exit_time is not None:
            return max(0.0, self.exit_time - self.entry_time)
        return 0.0


class QueueMonitor:
    """
    Tracks real-time queue length, service counter transactions, customer waiting times,
    and service turnover rates.
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

        # Counter service sessions
        self.active_counter_sessions: Dict[int, QueuePersonSession] = {}
        self.completed_service_sessions: List[QueuePersonSession] = []

    def reset(self) -> None:
        """Reset all queue and service history."""
        self.active_sessions.clear()
        self.completed_sessions.clear()
        self.active_counter_sessions.clear()
        self.completed_service_sessions.clear()

    def update(
        self,
        queue_tracks: List[Dict],
        counter_tracks: Optional[List[Dict]] = None,
        now: Optional[float] = None,
    ) -> Dict:
        """
        Update the queue and counter state with the current frame's tracked occupants.

        Args:
            queue_tracks: List of track dicts inside the queue ROI.
            counter_tracks: Optional list of track dicts inside the service counter ROI.
            now: Current timestamp (seconds). Uses time.time() if None.

        Returns:
            Dictionary of real-time queue and counter metrics.
        """
        if now is None:
            now = time.time()

        if counter_tracks is None:
            counter_tracks = []

        counter_track_ids = {trk["track_id"] for trk in counter_tracks if "track_id" in trk}
        # Avoid double-counting if a track is in both counter and queue: counter takes precedence
        queue_track_ids = {
            trk["track_id"] for trk in queue_tracks
            if "track_id" in trk and trk["track_id"] not in counter_track_ids
        }

        # -------------------------------------------------------------
        # 1. Update Service Counter Sessions
        # -------------------------------------------------------------
        for tid in counter_track_ids:
            if tid in self.active_counter_sessions:
                self.active_counter_sessions[tid].last_seen_time = now
            else:
                self.active_counter_sessions[tid] = QueuePersonSession(
                    track_id=tid, entry_time=now, zone="counter"
                )

        to_remove_counter = []
        for tid, sess in self.active_counter_sessions.items():
            if tid not in counter_track_ids:
                if (now - sess.last_seen_time) >= self.loss_grace_sec:
                    sess.exit_time = sess.last_seen_time
                    sess.is_active = False
                    duration = sess.exit_time - sess.entry_time
                    if duration >= self.min_dwell_sec:
                        self.completed_service_sessions.append(sess)
                        if len(self.completed_service_sessions) > self.max_history:
                            self.completed_service_sessions.pop(0)
                    to_remove_counter.append(tid)

        for tid in to_remove_counter:
            del self.active_counter_sessions[tid]

        # -------------------------------------------------------------
        # 2. Update Queue Line Sessions
        # -------------------------------------------------------------
        for tid in queue_track_ids:
            if tid in self.active_sessions:
                sess = self.active_sessions[tid]
                sess.last_seen_time = now
                if not sess.confirmed_waiting:
                    if (now - sess.entry_time) >= self.min_dwell_sec:
                        sess.confirmed_waiting = True
            else:
                new_sess = QueuePersonSession(track_id=tid, entry_time=now, zone="queue")
                if self.min_dwell_sec <= 0:
                    new_sess.confirmed_waiting = True
                self.active_sessions[tid] = new_sess

        to_remove_queue = []
        for tid, sess in self.active_sessions.items():
            if tid not in queue_track_ids:
                if (now - sess.last_seen_time) >= self.loss_grace_sec:
                    sess.exit_time = sess.last_seen_time
                    sess.is_active = False
                    duration = sess.exit_time - sess.entry_time
                    if sess.confirmed_waiting and duration >= self.min_dwell_sec:
                        self.completed_sessions.append(sess)
                        if len(self.completed_sessions) > self.max_history:
                            self.completed_sessions.pop(0)
                    to_remove_queue.append(tid)

        for tid in to_remove_queue:
            del self.active_sessions[tid]

        # -------------------------------------------------------------
        # 3. Calculate Aggregated Operational Metrics
        # -------------------------------------------------------------
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
                "zone": "queue"
            })

        # Counter service metrics
        current_service_time = 0.0
        for tid, sess in self.active_counter_sessions.items():
            elapsed = round(sess.elapsed_wait(now), 1)
            if elapsed > current_service_time:
                current_service_time = elapsed
            active_waiters.append({
                "track_id": tid,
                "elapsed_sec": elapsed,
                "confirmed": True,
                "zone": "counter"
            })

        # Average wait of completed queue sessions
        avg_completed_wait = 0.0
        if self.completed_sessions:
            total_time = sum(s.elapsed_wait(s.exit_time or now) for s in self.completed_sessions)
            avg_completed_wait = round(total_time / len(self.completed_sessions), 1)

        # Average service duration
        avg_service_time = 0.0
        if self.completed_service_sessions:
            total_svc_time = sum(s.elapsed_wait(s.exit_time or now) for s in self.completed_service_sessions)
            avg_service_time = round(total_svc_time / len(self.completed_service_sessions), 1)

        total_active_waiting = len(self.active_sessions) + len(self.active_counter_sessions)

        return {
            "current_queue_length": len(self.active_sessions),
            "queue_line_count": len(self.active_sessions),
            "counter_count": len(self.active_counter_sessions),
            "total_waiting_count": total_active_waiting,
            "confirmed_waiting_count": confirmed_count + len(self.active_counter_sessions),
            "longest_current_wait_sec": round(longest_current_wait, 1),
            "current_service_time_sec": round(current_service_time, 1),
            "avg_completed_wait_sec": avg_completed_wait,
            "avg_service_time_sec": avg_service_time,
            "total_completed_sessions": len(self.completed_sessions),
            "total_completed_services": len(self.completed_service_sessions),
            "active_waiters": sorted(active_waiters, key=lambda x: x["elapsed_sec"], reverse=True),
        }
