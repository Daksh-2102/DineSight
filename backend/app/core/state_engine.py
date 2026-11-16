"""
DineSight AI - Table-State Recognition Engine (Module 3B)
=========================================================
Implements the finite state machine (FSM) and temporal smoothing rules
for classifying dining table states.

Required States:
1. VACANT           : No patrons associated with the table.
2. OCCUPIED         : Patrons confirmed seated after dwell-time threshold.
3. WAITING_FOR_FOOD : Occupied, but no tableware/food items detected on table.
4. DINING           : Occupied, with tableware (cups, bowls) visually detected.

Key Engineering Features:
- Timestamp-based dwell time (prevents passers-by from triggering occupancy).
- Timestamp-based release period (grace period for temporary occlusion/motion).
- Temporal smoothing: Prevents rapid oscillation/flickering between states.
- Extensible heuristic: Treats tableware as estimated visual evidence rather than
  claiming absolute proof of dining.
- Ephemeral in-memory tracking: No database persistence in this module.
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class TableState(str, Enum):
    VACANT = "VACANT"
    OCCUPIED = "OCCUPIED"
    WAITING_FOR_FOOD = "WAITING_FOR_FOOD"
    DINING = "DINING"


class TableSession:
    """
    Maintains temporal state history and transition counters for a single dining table.
    """

    def __init__(
        self,
        table_id: str,
        dwell_time_sec: float = 2.0,
        release_time_sec: float = 2.5,
        activity_smoothing_sec: float = 2.0,
    ):
        self.table_id = table_id
        self.dwell_time_sec = float(dwell_time_sec)
        self.release_time_sec = float(release_time_sec)
        self.activity_smoothing_sec = float(activity_smoothing_sec)

        # Current stabilized state
        self.current_state: TableState = TableState.VACANT
        self.last_state_change_time: float = time.time()

        # Dwell & Release temporal candidate tracking
        self.candidate_state: Optional[TableState] = None
        self.candidate_start_time: float = 0.0

        # Occupant tracking
        self.confirmed_occupant_ids: Set[int] = set()
        self.current_frame_occupant_ids: Set[int] = set()
        self.last_seen_occupant_time: float = 0.0
        self.occupancy_start_time: Optional[float] = None

        # Tableware / Food activity evidence
        self.tableware_present: bool = False
        self.tableware_candidate_time: float = 0.0
        self.tableware_candidate_val: bool = False

    def reset(self) -> None:
        """Reset table state back to VACANT baseline."""
        self.current_state = TableState.VACANT
        self.last_state_change_time = time.time()
        self.candidate_state = None
        self.candidate_start_time = 0.0
        self.confirmed_occupant_ids.clear()
        self.current_frame_occupant_ids.clear()
        self.last_seen_occupant_time = 0.0
        self.occupancy_start_time = None
        self.tableware_present = False

    def update(
        self,
        occupant_tracks: List[Dict],
        tableware_detections: Optional[List[Dict]] = None,
        now: Optional[float] = None,
    ) -> TableState:
        """
        Process the current frame's observations and advance the state machine.
        """
        if now is None:
            now = time.time()

        # 1. Update current frame occupant IDs
        self.current_frame_occupant_ids = {trk["track_id"] for trk in occupant_tracks if "track_id" in trk}
        has_people_now = len(self.current_frame_occupant_ids) > 0

        # Check tableware evidence
        has_tableware_now = bool(tableware_detections and len(tableware_detections) > 0)

        # 2. Smooth tableware detection
        if has_tableware_now != self.tableware_candidate_val:
            self.tableware_candidate_val = has_tableware_now
            self.tableware_candidate_time = now
        elif (now - self.tableware_candidate_time) >= self.activity_smoothing_sec:
            self.tableware_present = self.tableware_candidate_val

        # 3. Finite State Machine Transitions
        if self.current_state == TableState.VACANT:
            if has_people_now:
                # A person is detected inside the table ROI.
                # Must stay inside for dwell_time_sec to confirm OCCUPIED.
                if self.candidate_state != TableState.OCCUPIED:
                    self.candidate_state = TableState.OCCUPIED
                    self.candidate_start_time = now
                elif (now - self.candidate_start_time) >= self.dwell_time_sec:
                    # Dwell time satisfied! Confirm transition to active occupied state
                    target_state = TableState.DINING if self.tableware_present else TableState.WAITING_FOR_FOOD
                    self._transition_to(target_state, now)
                    self.occupancy_start_time = now
                    self.confirmed_occupant_ids = set(self.current_frame_occupant_ids)
            else:
                # Table remains vacant
                self.candidate_state = None

        else:
            # Table is currently in an OCCUPIED-like state (OCCUPIED, WAITING_FOR_FOOD, or DINING)
            if has_people_now:
                # Patrons are actively present
                self.last_seen_occupant_time = now
                self.confirmed_occupant_ids.update(self.current_frame_occupant_ids)
                self.candidate_state = None  # Clear any pending release

                # Determine activity sub-state (Dining vs Waiting for Food)
                target_state = TableState.DINING if self.tableware_present else TableState.WAITING_FOR_FOOD
                if self.current_state in (TableState.OCCUPIED, TableState.WAITING_FOR_FOOD, TableState.DINING):
                    if self.current_state != target_state:
                        self._transition_to(target_state, now)

            else:
                # No patrons detected in this frame (could be temporary occlusion or departure)
                if self.candidate_state != TableState.VACANT:
                    self.candidate_state = TableState.VACANT
                    self.candidate_start_time = now
                elif (now - self.candidate_start_time) >= self.release_time_sec:
                    # Release grace period expired! Confirm table is VACANT
                    self._transition_to(TableState.VACANT, now)
                    self.occupancy_start_time = None
                    self.confirmed_occupant_ids.clear()
                    self.tableware_present = False

        return self.current_state

    def _transition_to(self, new_state: TableState, timestamp: float) -> None:
        """Internal helper to record state change and reset candidate."""
        self.current_state = new_state
        self.last_state_change_time = timestamp
        self.candidate_state = None

    def get_status(self, now: Optional[float] = None) -> Dict:
        """Return a structured summary of this table's operational state."""
        if now is None:
            now = time.time()

        occupied_duration = 0.0
        if self.occupancy_start_time is not None and self.current_state != TableState.VACANT:
            occupied_duration = round(now - self.occupancy_start_time, 1)

        state_age = round(now - self.last_state_change_time, 1)

        return {
            "table_id": self.table_id,
            "state": self.current_state.value,
            "occupant_count": len(self.current_frame_occupant_ids) if self.current_state != TableState.VACANT else 0,
            "occupant_ids": sorted(list(self.confirmed_occupant_ids if self.current_state != TableState.VACANT else [])),
            "occupied_duration_sec": occupied_duration,
            "state_age_sec": state_age,
            "tableware_detected": self.tableware_present,
            "is_occupied": self.current_state != TableState.VACANT,
        }


class TableStateEngine:
    """
    Manages all dining tables in the restaurant and coordinates their state machines.
    """

    def __init__(
        self,
        table_ids: List[str],
        dwell_time_sec: float = 2.0,
        release_time_sec: float = 2.5,
        activity_smoothing_sec: float = 2.0,
    ):
        self.tables: Dict[str, TableSession] = {
            t_id: TableSession(
                table_id=t_id,
                dwell_time_sec=dwell_time_sec,
                release_time_sec=release_time_sec,
                activity_smoothing_sec=activity_smoothing_sec,
            )
            for t_id in table_ids
        }

    def update(
        self,
        table_occupants: Dict[str, List[Dict]],
        table_tableware: Optional[Dict[str, List[Dict]]] = None,
        now: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Update all tables with current frame observations.

        Returns:
            Dictionary mapping table_id -> state string (e.g. {"table_1": "OCCUPIED", ...})
        """
        states = {}
        for t_id, session in self.tables.items():
            occupants = table_occupants.get(t_id, [])
            tableware = (table_tableware or {}).get(t_id, [])
            st = session.update(occupants, tableware, now=now)
            states[t_id] = st.value
        return states

    def get_all_statuses(self, now: Optional[float] = None) -> Dict[str, Dict]:
        """Get full status dictionary for all managed tables."""
        return {t_id: s.get_status(now) for t_id, s in self.tables.items()}

    def reset(self) -> None:
        """Reset all table state machines (e.g. on video rewind)."""
        for s in self.tables.values():
            s.reset()
