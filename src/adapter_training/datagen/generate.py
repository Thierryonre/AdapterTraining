"""DexMimicGen-style data generation (Part 2, steps 5-6), reimplemented on the MuJoCo handover env.

For each new episode:
  1. put the object at a random pose and pick a source demo
  2. for each phase, take the source's commanded gripper poses for that phase and re-express them
     relative to where the object is *now* (arms with ref "object"; arms that share the object in one
     phase share one transform), or replay them as-is (ref None)
  3. interpolate from the current gripper poses to the start of the transformed segment
  4. execute through IK + physics, keep the episode only if the handover succeeds
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from adapter_training.datagen.dataset import Episode, Recorder, SourceDemo
from adapter_training.datagen.env import SIDES, HandoverEnv, pose_matrix
from adapter_training.datagen.task import Phase


@dataclass
class GenConfig:
    # Object pose randomisation, relative to the chosen source demo's initial object pose.
    dx: tuple[float, float] = (-0.04, 0.04)
    dy: tuple[float, float] = (-0.05, 0.05)
    dyaw_deg: tuple[float, float] = (-20.0, 20.0)
    interp_steps: int = 8  # steps to move from the current pose to each transformed segment
    hold_steps: int = 10  # steps to hold at the end before checking success
    action_noise_m: float = 0.0  # optional position noise on commanded poses (MimicGen uses ~5 mm)


def sample_object_pose(base: np.ndarray, cfg: GenConfig, rng: np.random.Generator, height: float) -> np.ndarray:
    yaw0 = np.arctan2(base[1, 0], base[0, 0])
    yaw = yaw0 + np.radians(rng.uniform(*cfg.dyaw_deg))
    c, s = np.cos(yaw), np.sin(yaw)
    pos = base[:3, 3] + np.array([rng.uniform(*cfg.dx), rng.uniform(*cfg.dy), 0.0])
    pos[2] = height / 2
    return pose_matrix(pos, np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))


def interpolate(T0: np.ndarray, T1: np.ndarray, n: int) -> list[np.ndarray]:
    """n poses strictly after T0 up to and including T1 (linear position, slerp rotation)."""
    if n <= 0:
        return []
    slerp = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
    out = []
    for w in np.linspace(0, 1, n + 1)[1:]:
        out.append(pose_matrix((1 - w) * T0[:3, 3] + w * T1[:3, 3], slerp(w).as_matrix()))
    return out


def transformed_segment(src: SourceDemo, phase: Phase, object_now: np.ndarray) -> tuple[dict, np.ndarray]:
    a, b = src.boundaries[phase.name]
    # One transform per phase: maps the source object pose at the phase start onto the current one.
    transform = object_now @ np.linalg.inv(src.object_pose[a])
    seg = {}
    for s in SIDES:
        poses = src.target_pose[s][a:b]
        seg[s] = np.einsum("ij,tjk->tik", transform, poses) if phase.refs[s] == "object" else poses.copy()
    return seg, src.openings[a:b]


def generate_episode(
    env: HandoverEnv, src: SourceDemo, phases: list[Phase], object_pose: np.ndarray, cfg: GenConfig, rng
) -> tuple[Episode, bool, dict]:
    env.set_state(src.initial_state)  # same arm configuration as the source demo
    env.set_object_pose(object_pose)
    for _ in range(5):
        env._physics_step()
    rec = Recorder(env)
    current = {s: env.eef_pose(s) for s in SIDES}
    openings = {s: float(src.openings[0, i]) for i, s in enumerate(SIDES)}
    info = {}

    def run(targets_seq: list[dict], openings_seq: list[dict]) -> None:
        for targets, ops in zip(targets_seq, openings_seq):
            cmd = {}
            for s in SIDES:
                T = targets[s].copy()
                if cfg.action_noise_m:
                    T[:3, 3] += rng.normal(0, cfg.action_noise_m, 3)
                cmd[s] = T
            rec.before_step()
            arm = env.step_ee(cmd, ops)
            rec.after_command(cmd, arm, ops)

    boundaries = {}
    for phase in phases:
        seg, seg_open = transformed_segment(src, phase, env.object_pose())
        if len(seg_open) == 0:
            continue
        start = len(rec.ep)
        interp = {s: interpolate(current[s], seg[s][0], cfg.interp_steps) for s in SIDES}
        run(
            [{s: interp[s][i] for s in SIDES} for i in range(cfg.interp_steps)],
            [dict(openings) for _ in range(cfg.interp_steps)],
        )
        run(
            [{s: seg[s][t] for s in SIDES} for t in range(len(seg_open))],
            [{s: float(seg_open[t, i]) for i, s in enumerate(SIDES)} for t in range(len(seg_open))],
        )
        current = {s: seg[s][-1] for s in SIDES}
        openings = {s: float(seg_open[-1, i]) for i, s in enumerate(SIDES)}
        info[f"{phase.name}_grasped"] = {s: env.grasped(s) for s in SIDES}
        boundaries[phase.name] = [start, len(rec.ep)]

    run([dict(current) for _ in range(cfg.hold_steps)], [dict(openings) for _ in range(cfg.hold_steps)])
    boundaries[phases[-1].name][1] = len(rec.ep)  # the final hold belongs to the last phase
    info["boundaries"] = boundaries
    return rec.ep, env.is_success(), info
