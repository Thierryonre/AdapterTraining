"""Task structure for generation, inferred automatically from a demonstration.

DexMimicGen terms: a demo is split into phases (subtasks). In each phase an arm's motion is either
re-expressed relative to an object's current pose (ref = object name) or replayed as recorded
(ref = None). Arms that reference the same object in a phase share one transform, which keeps their
relative motion intact (DexMimicGen's coordination subtask, e.g. a handover).

Inference rules (no per-skill code):
  - phases end at gripper events (any arm closing or opening), in order
  - a closing arm's phase is relative to the object it grasps
  - an opening arm's phase is relative to the object the released one ends up on/in, or to the
    released object itself if the other arm still holds it (handover), or replayed (put down on table)
  - an arm already holding the object another arm references in that phase moves with it
  - success = the demo's final arrangement: who holds what, lifted or not, where placed objects ended
    up relative to what they were put on/in, and untouched objects still upright
"""

from dataclasses import asdict, dataclass

import numpy as np

SIDES = ("left", "right")
CLOSE_BELOW, OPEN_ABOVE = 0.3, 0.6
PLACE_RADIUS = 0.15  # a released object counts as placed on/in another object within this xy distance


@dataclass
class Phase:
    name: str
    end_event: str | None  # e.g. "left_close", "left_open"; None = until the end of the demo
    refs: dict  # side -> object name or None
    settle_s: float = 0.3  # the phase ends this long after the event (lets the gripper finish)


# The hand-written handover structure, kept for reference/tests. Inferred tasks look like this.
HANDOVER_PHASES = [
    Phase("left_grasp", "left_close", {"left": "bar", "right": None}),
    Phase("handover", "left_open", {"left": "bar", "right": "bar"}),
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


def event_sequence(openings: np.ndarray) -> list[tuple[int, str, str]]:
    """All gripper events of both arms in time order: (frame, side, "close"|"open")."""
    seq = []
    for k, side in enumerate(SIDES):
        for kind, frames in gripper_events(openings[:, k]).items():
            seq += [(t, side, kind) for t in frames]
    return sorted(seq)


def phase_boundaries(openings: np.ndarray, phases: list[Phase], fps: float) -> dict[str, tuple[int, int]]:
    """openings: (T, 2) left/right. Returns phase -> [start, end) frame indices."""
    events: dict[str, list[int]] = {}
    for t, side, kind in event_sequence(openings):
        events.setdefault(f"{side}_{kind}", []).append(t)
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


def _distance_to_box(point: np.ndarray, T_obj: np.ndarray, size: np.ndarray) -> float:
    local = T_obj[:3, :3].T @ (point - T_obj[:3, 3])
    return float(np.linalg.norm(np.maximum(np.abs(local) - size / 2, 0.0)))


def nearest_object(point: np.ndarray, poses: dict, sizes: dict, exclude=()) -> str | None:
    candidates = [(n, _distance_to_box(point, T, sizes[n])) for n, T in poses.items() if n not in exclude]
    return min(candidates, key=lambda c: c[1])[0] if candidates else None


def infer_task(openings: np.ndarray, eef: dict, object_poses: dict, sizes: dict, holders_at=None) -> dict:
    """Infer phases and the success check from one simulated replay.

    openings: (T, 2) commanded; eef: side -> (T, 4, 4); object_poses: name -> (T, 4, 4);
    sizes: name -> (3,); holders_at(t) -> {name: set(sides)} (from simulation contacts), optional.
    """
    events = event_sequence(openings)
    T = len(openings)
    held = {s: None for s in SIDES}
    last_release_ref: dict[str, str | None] = {}
    phases = []
    for i, (t, side, kind) in enumerate(events):
        poses_t = {n: P[t] for n, P in object_poses.items()}
        refs = {s: None for s in SIDES}
        if kind == "close":
            target = nearest_object(eef[side][t][:3, 3], poses_t, sizes)
            refs[side] = target
            held[side] = target
        else:
            released = held[side]
            other = [s for s in SIDES if s != side and held[s] == released and released is not None]
            if other:  # handing over: withdraw relative to the object the other hand now holds
                refs[side] = released
            elif released is not None:
                t_end = min(T - 1, t + 15)
                place_on = nearest_object(object_poses[released][t_end][:3, 3], poses_t, sizes, exclude=(released,))
                if place_on is not None:
                    d = np.linalg.norm(object_poses[place_on][t][:2, 3] - object_poses[released][t_end][:2, 3])
                    place_on = place_on if d < PLACE_RADIUS else None
                refs[side] = place_on
                last_release_ref[released] = place_on
            held[side] = None
        for s in SIDES:  # an arm holding the referenced object moves with it (coordination)
            if refs[s] is None and held[s] is not None and held[s] in refs.values():
                refs[s] = held[s]
        phases.append(Phase(f"p{i}_{side}_{kind}", f"{side}_{kind}", refs))
    phases.append(Phase(f"p{len(events)}_end", None, {s: None for s in SIDES}))

    final = {n: P[-1] for n, P in object_poses.items()}
    initial = {n: P[0] for n, P in object_poses.items()}
    final_holders = holders_at(T - 1) if holders_at else {n: {s for s in SIDES if held[s] == n} for n in object_poses}
    checks = {}
    for name in object_poses:
        rest_z = sizes[name][2] / 2
        moved = np.linalg.norm(final[name][:3, 3] - initial[name][:3, 3]) > 0.03
        check = {"holders": sorted(final_holders.get(name, set()))}
        if check["holders"]:
            check["lifted"] = bool(final[name][2, 3] > rest_z + 0.03)
        elif moved:
            ref = last_release_ref.get(name)
            frame = final[ref] if ref else np.eye(4)
            check["placed_relative_to"] = ref
            check["rel_pos"] = (np.linalg.inv(frame) @ np.append(final[name][:3, 3], 1.0))[:3].tolist()
            check["tolerance_m"] = 0.04
        else:
            check["stays_upright"] = True
        checks[name] = check
    return {"phases": [asdict(p) for p in phases], "success": checks, "events": [[t, s, k] for t, s, k in events]}


def phases_from_spec(spec: dict) -> list[Phase]:
    return [Phase(**p) for p in spec["phases"]]


def structure_signature(spec: dict) -> list:
    """What must match between source demos of one skill: event order and per-phase references."""
    return [(p["end_event"], tuple(sorted(p["refs"].items()))) for p in spec["phases"]]


def make_success_fn(spec: dict):
    """Callable(env) -> bool checking the demo's final arrangement."""
    checks = spec["success"]

    def success(env) -> bool:
        poses = env.object_poses()
        for name, check in checks.items():
            T = poses[name]
            if sorted(env.holders(name)) != check["holders"]:
                return False
            if check.get("lifted") and T[2, 3] <= env.object_size(name)[2] / 2 + 0.03:
                return False
            if "rel_pos" in check:
                ref = check["placed_relative_to"]
                frame = poses[ref] if ref else np.eye(4)
                rel = (np.linalg.inv(frame) @ np.append(T[:3, 3], 1.0))[:3]
                if np.linalg.norm(rel - np.asarray(check["rel_pos"])) > check["tolerance_m"]:
                    return False
            if check.get("stays_upright") and T[2, 2] < np.cos(np.radians(25)):
                return False
        return True

    return success


def check_events_present(openings: np.ndarray) -> str | None:
    """A demo needs at least one grasp to be generated from."""
    if not any(kind == "close" for _, _, kind in event_sequence(openings)):
        return "no grasp detected (the gripper never closes)"
    return None
