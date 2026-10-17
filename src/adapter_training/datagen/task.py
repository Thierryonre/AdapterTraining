"""Task description for generation: phases, which object each arm's motion is relative to, and how
phase boundaries are found from the gripper open/close events of a demo.

DexMimicGen terms: each phase is a subtask; an arm with ref "object" has its source motion
re-expressed relative to the object's current pose (object-centric), an arm with ref None replays
its source motion in the table frame. When both arms reference the object in the same phase they
share one transform, which keeps their relative motion intact (DexMimicGen's coordination subtask --
needed for the handover, where the right gripper grasps the object held by the left).
"""

from dataclasses import dataclass

import numpy as np

CLOSE_BELOW, OPEN_ABOVE = 0.3, 0.6


@dataclass
class Phase:
    name: str
    end_event: str | None  # e.g. "left_close", "left_open"; None = until the end of the demo
    refs: dict  # side -> "object" or None
    settle_s: float = 0.3  # the phase ends this long after the event (lets the gripper finish)


HANDOVER_PHASES = [
    Phase("left_grasp", "left_close", {"left": "object", "right": None}),
    Phase("handover", "left_open", {"left": "object", "right": "object"}),
    Phase("retreat", None, {"left": None, "right": None}),
]
HANDOVER_TASK = "Pass the bar from the left hand to the right hand."


def gripper_events(opening: np.ndarray) -> dict[str, list[int]]:
    """Frames where a gripper closes (opening drops below CLOSE_BELOW) or opens (rises above OPEN_ABOVE)."""
    events = {"close": [], "open": []}
    closed = opening[0] < CLOSE_BELOW
    for t, x in enumerate(opening):
        if not closed and x < CLOSE_BELOW:
            events["close"].append(t)
            closed = True
        elif closed and x > OPEN_ABOVE:
            events["open"].append(t)
            closed = False
    return events


def phase_boundaries(openings: np.ndarray, phases: list[Phase], fps: float) -> dict[str, tuple[int, int]]:
    """openings: (T, 2) left/right. Returns phase -> [start, end) frame indices."""
    events = {}
    for k, side in enumerate(("left", "right")):
        for kind, frames in gripper_events(openings[:, k]).items():
            events[f"{side}_{kind}"] = frames
    T = len(openings)
    bounds, start = {}, 0
    for phase in phases:
        if phase.end_event is None:
            end = T
        else:
            later = [t for t in events.get(phase.end_event, []) if t >= start]
            if not later:
                raise ValueError(f"no '{phase.end_event}' event after frame {start} (phase '{phase.name}')")
            end = min(T, later[0] + int(phase.settle_s * fps))
        bounds[phase.name] = (start, end)
        start = end
    return bounds


def check_handover_order(openings: np.ndarray) -> str | None:
    """The right gripper must close after the left closes and before the left opens."""
    left, right = gripper_events(openings[:, 0]), gripper_events(openings[:, 1])
    if not left["close"] or not right["close"] or not left["open"]:
        return "missing grasp/release events"
    lc = left["close"][0]
    lo = [t for t in left["open"] if t > lc]
    rc = [t for t in right["close"] if t > lc]
    if not lo or not rc or rc[0] > lo[0]:
        return "right gripper doesn't grasp while the left still holds the object"
    return None
