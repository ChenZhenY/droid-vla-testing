# ruff: noqa
import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from typing import List, Tuple, Iterable, Dict, Callable

import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image
import pandas as pd
import tqdm
import tyro
import h5py

from droid.robot_env import RobotEnv
from openpi_client import image_tools

faulthandler.enable()

# ========================= CONTROL / SAFETY PARAMS =========================
DROID_CONTROL_FREQUENCY = 15.0  # Hz
DT = 1.0 / DROID_CONTROL_FREQUENCY

# Workspace safety box (meters) in base frame
WORKSPACE_MIN = np.array([0.25, -0.45, 0.10])   # x,y,z
WORKSPACE_MAX = np.array([0.80,  0.45, 0.60])

# Per-tick motion limits
MAX_STEP_TRANSL = 0.010               # m per tick (15 Hz → ~0.15 m/s max)
MAX_STEP_ANG_RAD = np.deg2rad(2.0)    # rad per tick (~30 deg/s max)

# Trajectory timing targets (also clamped by per-tick limits)
LINEAR_SPEED = 0.05                    # m/s
ANGULAR_SPEED = np.deg2rad(15.0)       # rad/s
MIN_SEG_DURATION = 1.0                 # s
MAX_SEG_DURATION = 6.0                 # s

# Pattern amplitudes
RING_RADIUS = 0.08
DELTA_SMALL = 0.08
DELTA_Z_UP = 0.08
DELTA_Z_DOWN = 0.05

# ========================= IO HELPERS =========================
def save_rollout_to_hdf5(rollout_data_list, filename):
    with h5py.File(filename, 'w') as f:
        if not rollout_data_list:
            raise ValueError("rollout_data_list is empty")
        keys = rollout_data_list[0].keys()
        for key in keys:
            data = np.array([d[key] for d in rollout_data_list])
            if data.dtype.kind in {'U', 'O'}:
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset(key, data=data.astype('U'), dtype=dt)
            else:
                f.create_dataset(key, data=data)
    print(f"Saved rollout data to {filename}")

@dataclasses.dataclass
class Args:
    # Cameras
    left_camera_id: str = "33087938"
    wrist_camera_id: str = "18482824"
    external_camera: str = "left"  # choose from ["left", "right"]

    # Run style
    pattern: str = "cross"  # "cross" | "ring" | "grid"
    cycles: int = 1
    max_timesteps: int = 1800

    # Logging
    save_rollout_to_hdf5: bool = False

# ========================= ROTATION / POSE MATH =========================
DROID_QUAT_ORDER = "wxyz"  # "wxyz" or "xyzw"

import numpy as np

def _aa_to_quat_xyzw(aa: np.ndarray) -> np.ndarray:
    """Axis-angle vector (ax,ay,az) → quaternion [x,y,z,w]."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.array([0., 0., 0., 1.], dtype=np.float64)
    axis = aa / theta
    s = np.sin(theta / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(theta/2.0)], dtype=np.float64)

def _quat_xyzw_to_aa(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] → axis-angle (ax,ay,az)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w = np.clip(q[3], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    s = np.sqrt(max(1.0 - w*w, 0.0))
    axis = q[:3] / (s + 1e-12)
    return axis * theta

def _quat_xyzw_to_droid(q_xyzw: np.ndarray) -> np.ndarray:
    """[x,y,z,w] → DROID order."""
    qx, qy, qz, qw = np.asarray(q_xyzw, dtype=np.float64)
    if DROID_QUAT_ORDER == "wxyz":
        return np.array([qw, qx, qy, qz], dtype=np.float64)
    elif DROID_QUAT_ORDER == "xyzw":
        return np.array([qx, qy, qz, qw], dtype=np.float64)
    else:
        raise ValueError(f"Unknown DROID_QUAT_ORDER={DROID_QUAT_ORDER}")

def _quat_droid_to_xyzw(q_droid: np.ndarray) -> np.ndarray:
    """DROID order → [x,y,z,w]."""
    q_droid = np.asarray(q_droid, dtype=np.float64)
    if DROID_QUAT_ORDER == "wxyz":
        qw, qx, qy, qz = q_droid
        return np.array([qx, qy, qz, qw], dtype=np.float64)
    elif DROID_QUAT_ORDER == "xyzw":
        qx, qy, qz, qw = q_droid
        return np.array([qx, qy, qz, qw], dtype=np.float64)
    else:
        raise ValueError(f"Unknown DROID_QUAT_ORDER={DROID_QUAT_ORDER}")

def pose6_to_pose7_droid(pose6: np.ndarray) -> np.ndarray:
    """
    [x,y,z, ax,ay,az]  →  [x,y,z, quat(4) in DROID order]
    """
    pose6 = np.asarray(pose6, dtype=np.float64)
    p = pose6[:3]
    aa = pose6[3:]
    q_xyzw = _aa_to_quat_xyzw(aa)
    q_droid = _quat_xyzw_to_droid(q_xyzw)
    return np.concatenate([p, q_droid], axis=0)

def pose7_droid_to_pose6(pose7: np.ndarray) -> np.ndarray:
    """
    [x,y,z, quat(4) in DROID order]  →  [x,y,z, ax,ay,az]
    Handy if your observation already comes as 7D quaternion.
    """
    pose7 = np.asarray(pose7, dtype=np.float64)
    p = pose7[:3]
    q_xyzw = _quat_droid_to_xyzw(pose7[3:])
    aa = _quat_xyzw_to_aa(q_xyzw)
    return np.concatenate([p, aa], axis=0)

def _aa_norm(aa: np.ndarray) -> float:
    return float(np.linalg.norm(aa))

def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> quaternion [x,y,z,w]."""
    theta = _aa_norm(aa)
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = aa / theta
    s = np.sin(theta / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(theta/2.0)], dtype=np.float64)

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q) + 1e-12)

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)

def _quat_conj(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)

def _quat_to_aa(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q.astype(np.float64))
    w = np.clip(q[3], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    s = np.sqrt(max(1.0 - w*w, 0.0))
    axis = q[:3] / (s + 1e-12)
    return axis * theta

def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        q = q0 + t * (q1 - q0)
        return _quat_normalize(q)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1

def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    dq = _quat_mul(_quat_normalize(q1), _quat_conj(_quat_normalize(q0)))
    ang = 2.0 * np.arccos(np.clip(dq[3], -1.0, 1.0))
    if ang > np.pi:
        ang = 2*np.pi - ang
    return float(ang)

def _delta_quat_from_to(q_from: np.ndarray, q_to: np.ndarray) -> Tuple[np.ndarray, float]:
    dq = _quat_mul(q_to, _quat_conj(q_from))
    ang = 2.0 * np.arccos(np.clip(dq[3], -1.0, 1.0))
    if ang > np.pi:
        # shortest path
        dq = -dq
        ang = 2*np.pi - ang
    return _quat_normalize(dq), float(ang)

# ========================= TRAJECTORY UTILITIES =========================
def _min_jerk_scalar(alpha: float) -> float:
    return (10*alpha**3 - 15*alpha**4 + 6*alpha**5)

def _plan_segment_duration(dx: float, dtheta: float) -> float:
    t_lin = abs(dx) / max(LINEAR_SPEED, 1e-6)
    t_ang = abs(dtheta) / max(ANGULAR_SPEED, 1e-6)
    T = max(t_lin, t_ang, MIN_SEG_DURATION)
    return float(min(T, MAX_SEG_DURATION))

def _clamp_workspace(p: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(p, WORKSPACE_MIN), WORKSPACE_MAX)

def _limit_step_6d(p_curr: np.ndarray, aa_curr: np.ndarray,
                   p_next: np.ndarray, aa_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp translation and rotation between current and next (one tick)."""
    # Translation limit
    dp = p_next - p_curr
    dist = np.linalg.norm(dp)
    if dist > MAX_STEP_TRANSL and dist > 1e-12:
        dp = dp * (MAX_STEP_TRANSL / dist)
        p_next = p_curr + dp

    # Rotation limit via delta quaternion
    q_curr = _aa_to_quat(aa_curr)
    q_next = _aa_to_quat(aa_next)
    dq, ang = _delta_quat_from_to(q_curr, q_next)
    if ang > MAX_STEP_ANG_RAD:
        # take a partial step toward target
        axis = dq[:3]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            q_limited = q_curr
        else:
            axis = axis / axis_norm
            s = np.sin(MAX_STEP_ANG_RAD/2.0)
            dq_step = np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(MAX_STEP_ANG_RAD/2.0)], dtype=np.float64)
            q_limited = _quat_mul(dq_step, q_curr)
        aa_next = _quat_to_aa(q_limited)

    return _clamp_workspace(p_next), aa_next

def _generate_minjerk_cartesian_6d(
    p0: np.ndarray, aa0: np.ndarray, p1: np.ndarray, aa1: np.ndarray
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Min-jerk in position + slerp in orientation (converted to axis-angle per sample)."""
    q0 = _aa_to_quat(aa0)
    q1 = _aa_to_quat(aa1)
    T = _plan_segment_duration(np.linalg.norm(p1 - p0), _quat_angle(q0, q1))
    N = max(2, int(np.ceil(T / DT)))
    for k in range(1, N + 1):
        a = k / float(N)
        s = _min_jerk_scalar(a)
        p_t = p0 + s * (p1 - p0)
        q_t = _quat_slerp(q0, q1, s)
        aa_t = _quat_to_aa(q_t)
        yield p_t, aa_t

# ========================= PATTERN GENERATORS =========================
def pattern_cross() -> List[Tuple[np.ndarray, np.ndarray]]:
    """List of (delta_xyz, delta_aa). delta_aa is axis-angle vector in base frame."""
    dels = []
    # ±X, ±Y translations
    for dx, dy in [(+DELTA_SMALL, 0), (-DELTA_SMALL, 0), (0, +DELTA_SMALL), (0, -DELTA_SMALL)]:
        dels.append((np.array([dx, dy, 0.0]), np.zeros(3)))
    # Up / down
    dels += [
        (np.array([0.0, 0.0, +DELTA_Z_UP]),   np.zeros(3)),
        (np.array([0.0, 0.0, -DELTA_Z_DOWN]), np.zeros(3)),
    ]
    # Slight yaw left/right for parallax (about base Z)
    yaw = np.deg2rad(15.0)
    dels += [
        (np.array([+0.06, +0.06, 0.0]), np.array([0.0, 0.0, +yaw])),
        (np.array([-0.06, -0.06, 0.0]), np.array([0.0, 0.0, -yaw])),
    ]
    return dels

def pattern_ring(num_pts: int = 12) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Circle in XY plane, constant z, small continuous yaw."""
    dels = []
    yaw_per = np.deg2rad(360.0 / num_pts) * 0.25
    for i in range(num_pts):
        th = 2*np.pi * i / num_pts
        dx = RING_RADIUS * np.cos(th)
        dy = RING_RADIUS * np.sin(th)
        dels.append((np.array([dx, dy, 0.0]), np.array([0.0, 0.0, yaw_per])))
    return dels

def pattern_grid(spacing: float = 0.06) -> List[Tuple[np.ndarray, np.ndarray]]:
    """3x3 grid in XY around current pose, constant z, no rotation."""
    dels = []
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            dels.append((np.array([ix*spacing, iy*spacing, 0.0]), np.zeros(3)))
    return dels

PATTERNS: Dict[str, Callable[[], List[Tuple[np.ndarray, np.ndarray]]]] = {
    "cross": pattern_cross,
    "ring": lambda: pattern_ring(12),
    "grid": pattern_grid,
}

# ========================= OBS / ACTION (6D) =========================
def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    """
    Expect obs_dict['robot_state']['cartesian_position'] to be 6D:
    [x, y, z, ax, ay, az] where [ax, ay, az] is axis-angle (base frame).
    """
    print("Extracting observations...")
    print(f"Keys in obs_dict: {list(obs_dict.keys())}")

    image_observations = obs_dict["image"]
    left_image, wrist_image = None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop alpha and convert BGR->RGB
    left_image = left_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"], dtype=np.float64)  # 6D now
    assert cartesian_position.shape[-1] == 6, "cartesian_position must be 6D [x,y,z,ax,ay,az]"
    joint_position = np.array(robot_state["joint_positions"], dtype=np.float64)
    gripper_position = np.array([robot_state["gripper_position"]], dtype=np.float64)

    if save_to_disk:
        combined = Image.fromarray(np.concatenate([left_image, wrist_image], axis=1))
        combined.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }

def _pose6_split(pose6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pose6[:3], dtype=np.float64)
    aa = np.asarray(pose6[3:], dtype=np.float64)
    return p, aa

def _pose6_pack(p: np.ndarray, aa: np.ndarray) -> np.ndarray:
    return np.concatenate([p.astype(np.float64), aa.astype(np.float64)], axis=0)

def _apply_delta_pose6(base_pose6: np.ndarray, d_xyz: np.ndarray, d_aa: np.ndarray) -> np.ndarray:
    """
    Apply translation in base frame and compose orientation by left-multiplying
    the delta axis-angle (also in base frame).
    """
    p0, aa0 = _pose6_split(base_pose6)
    p1 = p0 + d_xyz

    q0 = _aa_to_quat(aa0)
    q_delta = _aa_to_quat(d_aa)
    q1 = _quat_mul(q_delta, q0)  # base-frame delta then current
    aa1 = _quat_to_aa(q1)
    return _pose6_pack(p1, aa1)

# ========================= MAIN LOOP =========================
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)
    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt

def _plan_pattern_absolute_sequence(current_pose6: np.ndarray, pattern_name: str, cycles: int) -> List[np.ndarray]:
    if pattern_name not in PATTERNS:
        raise ValueError(f"Unknown pattern '{pattern_name}'. Choose from {list(PATTERNS.keys())}")
    deltas = PATTERNS[pattern_name]()
    seq: List[np.ndarray] = []
    pose = current_pose6.copy()
    for _ in range(cycles):
        for d_xyz, d_aa in deltas:
            goal = _apply_delta_pose6(pose, d_xyz, d_aa)
            gp, gaa = _pose6_split(goal)
            gp = _clamp_workspace(gp)
            goal = _pose6_pack(gp, gaa)
            seq.append(goal)
            pose = goal.copy()
    return seq

def main(args: Args):
    assert args.external_camera in ["left", "right"], "args.external_camera must be 'left' or 'right'"
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    print("Created the droid env (6D cartesian position + axis-angle)!")

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = None
        while instruction is None:
            instruction = input("Enter instruction (type 'sweep' to run camera sweep, or 'reset'): ").strip().lower()
            if instruction == 'reset':
                env.reset()
                instruction = None

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        save_rollout_data_list = []
        video = []

        # Initial observation
        obs = _extract_observation(args, env.get_observation(), save_to_disk=True)
        curr_pose6 = obs["cartesian_position"].copy()

        # Goals
        if instruction == "sweep":
            goals = _plan_pattern_absolute_sequence(curr_pose6, args.pattern, args.cycles)
        else:
            print("Unknown instruction, running default 'cross' once.")
            goals = _plan_pattern_absolute_sequence(curr_pose6, "cross", 1)

        bar = tqdm.tqdm(total=len(goals), desc="Segments")
        t_total = 0
        try:
            for goal_idx, goal_pose6 in enumerate(goals):
                # Refresh current pose for each segment
                curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=(goal_idx == 0))
                p0, aa0 = _pose6_split(curr_obs["cartesian_position"])
                p1, aa1 = _pose6_split(goal_pose6)

                # Generate min-jerk position + slerp orientation, then convert to AA
                for p_des, aa_des in _generate_minjerk_cartesian_6d(p0, aa0, p1, aa1):
                    # Enforce per-tick limits against current measured pose
                    p_curr, aa_curr = _pose6_split(curr_obs["cartesian_position"])
                    p_limited, aa_limited = _limit_step_6d(p_curr, aa_curr, p_des, aa_des)
                    action6 = _pose6_pack(p_limited, aa_limited)  # 6D absolute
                    action7 = pose6_to_pose7_droid(action6)   # -> [x,y,z, quat(4) in DROID order]
                    assert action7.shape[-1] == 7

                    # Send action (absolute 6D pose)
                    env.step(action7)

                    # Log + video
                    left_img = curr_obs[f"{args.external_camera}_image"]
                    video.append(left_img)
                    save_rollout_data_list.append({
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(left_img, 224, 224),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "target/pose6": action6.copy(),
                        "meta/segment_index": goal_idx,
                    })

                    # Keep loop at ~15 Hz
                    t0 = time.time()
                    elapsed = time.time() - t0
                    if elapsed < DT:
                        time.sleep(DT - elapsed)
                    t_total += 1

                    # Update obs
                    curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=False)

                bar.update(1)
        except KeyboardInterrupt:
            print("Interrupted by user. Stopping sweep.")
        finally:
            bar.close()

        # Save media & metrics
        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        if args.save_rollout_to_hdf5:
            save_rollout_to_hdf5(save_rollout_data_list, save_filename + ".hdf5")

        # Simple success prompt
        success: str | float | None = None
        while not isinstance(success, float):
            success = input("Did the sweep succeed? (y=100, n=0, or 0–100): ")
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0
            else:
                success = float(success) / 100.0
            if not (0.0 <= success <= 1.0):
                print(f"Success must be in [0,100]. Got {success*100:.1f}")

        df = df.append({"success": success, "duration": t_total, "video_filename": save_filename}, ignore_index=True)
        if input("Run another sweep? (y/n) ").strip().lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{ts}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)


# # ruff: noqa
# import contextlib
# import dataclasses
# import datetime
# import faulthandler
# import os
# import signal
# import time
# from typing import List, Tuple, Iterable, Dict, Callable

# import numpy as np
# from moviepy.editor import ImageSequenceClip
# from PIL import Image
# import pandas as pd
# import tqdm
# import tyro
# import h5py

# from droid.robot_env import RobotEnv  # your env
# from openpi_client import image_tools

# faulthandler.enable()

# # ========================= CONTROL / SAFETY PARAMS =========================
# DROID_CONTROL_FREQUENCY = 15.0  # Hz
# DT = 1.0 / DROID_CONTROL_FREQUENCY

# # Workspace safety box (meters) in base frame
# WORKSPACE_MIN = np.array([0.25, -0.45, 0.10])   # x,y,z
# WORKSPACE_MAX = np.array([0.80,  0.45, 0.60])

# # Motion limits
# MAX_STEP_TRANSL = 0.010  # m per tick (15 Hz → ~0.15 m/s max if saturated)
# MAX_STEP_ANG_RAD = np.deg2rad(2.0)  # rad per tick (≈30 deg/s max if saturated)

# # Trajectory limits / speeds
# LINEAR_SPEED = 0.05       # m/s target for timing
# ANGULAR_SPEED = np.deg2rad(15.0)  # rad/s target for timing
# MIN_SEG_DURATION = 1.0    # s, minimum segment time
# MAX_SEG_DURATION = 6.0    # s, maximum segment time

# # Small ring radius (m) and grid/cross deltas (m)
# RING_RADIUS = 0.08
# DELTA_SMALL = 0.08
# DELTA_Z_UP = 0.08
# DELTA_Z_DOWN = 0.05

# # ========================= IO HELPERS (unchanged) =========================
# def save_rollout_to_hdf5(rollout_data_list, filename):
#     with h5py.File(filename, 'w') as f:
#         if not rollout_data_list:
#             raise ValueError("rollout_data_list is empty")
#         keys = rollout_data_list[0].keys()
#         for key in keys:
#             data = np.array([d[key] for d in rollout_data_list])
#             if data.dtype.kind in {'U', 'O'}:
#                 dt = h5py.string_dtype(encoding='utf-8')
#                 f.create_dataset(key, data=data.astype('U'), dtype=dt)
#             else:
#                 f.create_dataset(key, data=data)
#     print(f"Saved rollout data to {filename}")

# @dataclasses.dataclass
# class Args:
#     # Cameras
#     left_camera_id: str = "33087938"
#     wrist_camera_id: str = "18482824"
#     external_camera: str = None  # choose from ["left", "right"] (we use "left")

#     # Run style
#     pattern: str = "cross"  # "cross" | "ring" | "grid"
#     cycles: int = 1         # how many times to repeat the pattern
#     max_timesteps: int = 1800
#     open_loop_horizon: int = 8  # unused here, but kept for compatibility

#     # Remote (kept in case you still want to log policy requests)
#     remote_host: str = "0.0.0.0"
#     remote_port: int = 8000

#     save_rollout_to_hdf5: bool = False
#     torque_bias: bool = False  # not used

# # ========================= QUAT / POSE MATH =========================
# # ===== Quaternion order adapter for DROID =====
# # DROID commonly uses [qw, qx, qy, qz]  (a.k.a. "wxyz"). If your setup is [qx,qy,qz,qw],
# # change the line below to DROID_QUAT_ORDER = "xyzw".
# DROID_QUAT_ORDER = "wxyz"  # <-- CHANGE THIS IF NEEDED: "wxyz" or "xyzw"

# def _droid_to_xyzw(q_droid: np.ndarray) -> np.ndarray:
#     """Convert DROID quaternion order to internal [x,y,z,w]."""
#     q_droid = np.asarray(q_droid, dtype=np.float64)
#     if DROID_QUAT_ORDER == "wxyz":
#         qw, qx, qy, qz = q_droid
#         return np.array([qx, qy, qz, qw], dtype=np.float64)
#     elif DROID_QUAT_ORDER == "xyzw":
#         qx, qy, qz, qw = q_droid
#         return np.array([qx, qy, qz, qw], dtype=np.float64)
#     else:
#         raise ValueError(f"Unknown DROID_QUAT_ORDER={DROID_QUAT_ORDER}")

# def _xyzw_to_droid(q_xyzw: np.ndarray) -> np.ndarray:
#     """Convert internal [x,y,z,w] back to DROID order for commands."""
#     qx, qy, qz, qw = np.asarray(q_xyzw, dtype=np.float64)
#     if DROID_QUAT_ORDER == "wxyz":
#         return np.array([qw, qx, qy, qz], dtype=np.float64)
#     elif DROID_QUAT_ORDER == "xyzw":
#         return np.array([qx, qy, qz, qw], dtype=np.float64)
#     else:
#         raise ValueError(f"Unknown DROID_QUAT_ORDER={DROID_QUAT_ORDER}")

# def _quat_normalize(q: np.ndarray) -> np.ndarray:
#     return q / (np.linalg.norm(q) + 1e-12)

# def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
#     # xyzw
#     x1, y1, z1, w1 = q1
#     x2, y2, z2, w2 = q2
#     return np.array([
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2,
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#     ], dtype=np.float64)

# def _quat_conj(q: np.ndarray) -> np.ndarray:
#     x, y, z, w = q
#     return np.array([-x, -y, -z, w], dtype=np.float64)

# def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
#     # Ensure shortest path
#     q0 = _quat_normalize(q0.astype(np.float64))
#     q1 = _quat_normalize(q1.astype(np.float64))
#     dot = np.dot(q0, q1)
#     if dot < 0.0:
#         q1 = -q1
#         dot = -dot
#     DOT_THRESHOLD = 0.9995
#     if dot > DOT_THRESHOLD:
#         # Linear interpolation to avoid numerical issues
#         q = q0 + t * (q1 - q0)
#         return _quat_normalize(q)
#     theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
#     sin_theta_0 = np.sin(theta_0)
#     theta = theta_0 * t
#     s0 = np.sin(theta_0 - theta) / sin_theta_0
#     s1 = np.sin(theta) / sin_theta_0
#     return s0 * q0 + s1 * q1

# def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
#     axis = axis.astype(np.float64)
#     axis = axis / (np.linalg.norm(axis) + 1e-12)
#     s = np.sin(angle / 2.0)
#     return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2.0)], dtype=np.float64)

# # ========================= TRAJECTORY UTILITIES =========================
# def _min_jerk_scalar(alpha: float) -> float:
#     # 10 a^3 - 15 a^4 + 6 a^5, a in [0,1]
#     return (10*alpha**3 - 15*alpha**4 + 6*alpha**5)

# def _plan_segment_duration(dx: float, dtheta: float) -> float:
#     # choose the slower of linear/angular times, clamp
#     t_lin = abs(dx) / max(LINEAR_SPEED, 1e-6)
#     t_ang = abs(dtheta) / max(ANGULAR_SPEED, 1e-6)
#     T = max(t_lin, t_ang, MIN_SEG_DURATION)
#     return float(min(T, MAX_SEG_DURATION))

# def _clamp_workspace(p: np.ndarray) -> np.ndarray:
#     return np.minimum(np.maximum(p, WORKSPACE_MIN), WORKSPACE_MAX)

# def _limit_step(curr_p: np.ndarray, next_p: np.ndarray,
#                 curr_q: np.ndarray, next_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     # Limit translation per tick
#     dp = next_p - curr_p
#     dist = np.linalg.norm(dp)
#     if dist > MAX_STEP_TRANSL:
#         dp = dp * (MAX_STEP_TRANSL / dist)
#         next_p = curr_p + dp
#     # Limit rotation per tick (via delta quaternion)
#     dq = _quat_mul(next_q, _quat_conj(curr_q))
#     # map to angle
#     angle = 2.0 * np.arccos(np.clip(dq[3], -1.0, 1.0))
#     if angle > np.pi:
#         angle = 2*np.pi - angle
#     if angle > MAX_STEP_ANG_RAD:
#         scale = MAX_STEP_ANG_RAD / angle
#         # scale angle around axis
#         axis = dq[:3]
#         axis_norm = np.linalg.norm(axis)
#         if axis_norm < 1e-9:
#             # no-op if numerical
#             scaled = curr_q
#         else:
#             axis = axis / axis_norm
#             scaled_dq = _quat_from_axis_angle(axis, MAX_STEP_ANG_RAD)  # shortest
#             scaled = _quat_mul(scaled_dq, curr_q)
#         next_q = _quat_normalize(scaled)
#     return _clamp_workspace(next_p), _quat_normalize(next_q)

# def _generate_minjerk_cartesian(
#     p0: np.ndarray, q0: np.ndarray, p1: np.ndarray, q1: np.ndarray
# ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
#     # total deltas
#     T = _plan_segment_duration(np.linalg.norm(p1 - p0), _quat_angle(q0, q1))
#     N = max(2, int(np.ceil(T / DT)))
#     for k in range(1, N + 1):
#         a = k / float(N)
#         s = _min_jerk_scalar(a)
#         p_t = p0 + s * (p1 - p0)
#         q_t = _quat_slerp(q0, q1, s)
#         yield p_t, q_t

# def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
#     dq = _quat_mul(_quat_normalize(q1), _quat_conj(_quat_normalize(q0)))
#     ang = 2.0 * np.arccos(np.clip(dq[3], -1.0, 1.0))
#     if ang > np.pi:
#         ang = 2*np.pi - ang
#     return float(ang)

# # ========================= PATTERN GENERATORS =========================
# def pattern_cross() -> List[Tuple[np.ndarray, np.ndarray]]:
#     """List of (delta_xyz, delta_axisangle) yaw around base z."""
#     dels = []
#     # ±X, ±Y
#     for dx, dy in [(+DELTA_SMALL, 0), (-DELTA_SMALL, 0), (0, +DELTA_SMALL), (0, -DELTA_SMALL)]:
#         dels.append((np.array([dx, dy, 0.0]), np.array([0, 0, 0.0])))
#     # Up / down small
#     dels += [
#         (np.array([0.0, 0.0, +DELTA_Z_UP]),   np.array([0, 0, 0.0])),
#         (np.array([0.0, 0.0, -DELTA_Z_DOWN]), np.array([0, 0, 0.0])),
#     ]
#     # slight yaw left/right for parallax
#     yaw = np.deg2rad(15.0)
#     dels += [
#         (np.array([+0.06, +0.06, 0.0]), np.array([0, 0, +yaw])),
#         (np.array([-0.06, -0.06, 0.0]), np.array([0, 0, -yaw])),
#     ]
#     return dels

# def pattern_ring(num_pts: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
#     """Circle in XY plane around current pose, keep z and add small yaw."""
#     dels = []
#     yaw_per = np.deg2rad(360.0 / num_pts) * 0.25  # modest total yaw
#     for i in range(num_pts):
#         th = 2*np.pi * i / num_pts
#         dx = RING_RADIUS * np.cos(th)
#         dy = RING_RADIUS * np.sin(th)
#         dels.append((np.array([dx, dy, 0.0]), np.array([0, 0, yaw_per])))
#     return dels

# def pattern_grid(spacing: float = 0.06) -> List[Tuple[np.ndarray, np.ndarray]]:
#     """3x3 grid in XY around current pose (centered), constant z."""
#     dels = []
#     for ix in [-1, 0, 1]:
#         for iy in [-1, 0, 1]:
#             dels.append((np.array([ix*spacing, iy*spacing, 0.0]), np.array([0, 0, 0.0])))
#     return dels

# PATTERNS: Dict[str, Callable[[], List[Tuple[np.ndarray, np.ndarray]]]] = {
#     "cross": pattern_cross,
#     "ring": lambda: pattern_ring(12),
#     "grid": pattern_grid,
# }

# # ========================= OBS & ACTION =========================
# def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
#     print("Extracting observations...")
#     print(f"Keys in obs_dict: {list(obs_dict.keys())}")
#     image_observations = obs_dict["image"]
#     left_image, wrist_image = None, None
#     for key in image_observations:
#         if args.left_camera_id in key and "left" in key:
#             left_image = image_observations[key]
#         elif args.wrist_camera_id in key and "left" in key:
#             wrist_image = image_observations[key]

#     left_image = left_image[..., :3][..., ::-1]
#     wrist_image = wrist_image[..., :3][..., ::-1]

#     robot_state = obs_dict["robot_state"]
#     cartesian_position = np.array(robot_state["cartesian_position"], dtype=np.float64)  # [x,y,z,qx,qy,qz,qw]
#     joint_position = np.array(robot_state["joint_positions"], dtype=np.float64)
#     gripper_position = np.array([robot_state["gripper_position"]], dtype=np.float64)

#     if save_to_disk:
#         combined = Image.fromarray(np.concatenate([left_image, wrist_image], axis=1))
#         combined.save("robot_camera_views.png")

#     return {
#         "left_image": left_image,
#         "wrist_image": wrist_image,
#         "cartesian_position": cartesian_position,
#         "joint_position": joint_position,
#         "gripper_position": gripper_position,
#     }

# def _pose_split(pose7: np.ndarray):
#     """
#     DROID pose7 is [x, y, z, quat...] where quat order follows DROID_QUAT_ORDER.
#     Internally we always compute with [x,y,z,w] (xyzw).
#     """
#     p = np.asarray(pose7[:3], dtype=np.float64)
#     q_droid = np.asarray(pose7[3:], dtype=np.float64)
#     q_xyzw = _droid_to_xyzw(q_droid)
#     return p, _quat_normalize(q_xyzw)

# def _pose_pack(p: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
#     """
#     Convert internal [x,y,z,w] back to DROID quaternion order when emitting actions.
#     """
#     q_droid = _xyzw_to_droid(_quat_normalize(q_xyzw))
#     return np.concatenate([np.asarray(p, dtype=np.float64), q_droid], axis=0)

# def _apply_delta_pose(base_pose: np.ndarray, d_xyz: np.ndarray, d_rpy_axisangle: np.ndarray) -> np.ndarray:
#     """Apply delta in the **base frame**: position + yaw (z-axis) by default."""
    
#     p0, q0 = _pose_split(base_pose)
#     p1 = p0 + d_xyz
#     # Only yaw around base z unless you want roll/pitch—d_rpy_axisangle is [rx,ry,rz] radians about base axes
#     q_delta = _quat_from_axis_angle(np.array([1,0,0]), d_rpy_axisangle[0])
#     q_delta = _quat_mul(_quat_from_axis_angle(np.array([0,1,0]), d_rpy_axisangle[1]), q_delta)
#     q_delta = _quat_mul(_quat_from_axis_angle(np.array([0,0,1]), d_rpy_axisangle[2]), q_delta)
#     q1 = _quat_mul(q_delta, q0)
#     return _pose_pack(p1, q1)

# # ========================= MAIN LOOP =========================
# @contextlib.contextmanager
# def prevent_keyboard_interrupt():
#     interrupted = False
#     original_handler = signal.getsignal(signal.SIGINT)
#     def handler(signum, frame):
#         nonlocal interrupted
#         interrupted = True
#     signal.signal(signal.SIGINT, handler)
#     try:
#         yield
#     finally:
#         signal.signal(signal.SIGINT, original_handler)
#         if interrupted:
#             raise KeyboardInterrupt

# def _plan_pattern_absolute_sequence(current_pose: np.ndarray, pattern_name: str, cycles: int) -> List[np.ndarray]:
#     if pattern_name not in PATTERNS:
#         raise ValueError(f"Unknown pattern '{pattern_name}'. Choose from {list(PATTERNS.keys())}")
#     deltas = PATTERNS[pattern_name]()
#     seq: List[np.ndarray] = []
#     pose = current_pose.copy()
#     for _ in range(cycles):
#         for d_xyz, d_rpy in deltas:
#             goal = _apply_delta_pose(pose, d_xyz, d_rpy)
#             # clamp goal position to workspace
#             gp, gq = _pose_split(goal)
#             gp = _clamp_workspace(gp)
#             goal = _pose_pack(gp, gq)
#             seq.append(goal)
#             # next segment starts from the last goal
#             pose = goal.copy()
#     return seq

# def main(args: Args):
#     assert args.external_camera in ["left", "right"], "args.external_camera must be 'left' or 'right'"
#     env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
#     print("Created the droid env!")

#     df = pd.DataFrame(columns=["success", "duration", "video_filename"])

#     while True:
#         instruction = None
#         while instruction is None:
#             instruction = input("Enter instruction (type 'sweep' to run camera sweep, or 'reset'): ").strip().lower()
#             if instruction == 'reset':
#                 env.reset()
#                 instruction = None

#         # Prepare run
#         timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
#         save_rollout_data_list = []
#         video = []

#         # === Get initial observation ===
#         obs = _extract_observation(args, env.get_observation(), save_to_disk=True)
#         curr_pose = obs["cartesian_position"].copy()

#         print(f"current pose {curr_pose.shape}")

#         # === Build absolute goals from pattern ===
#         if instruction == "sweep":
#             goals = _plan_pattern_absolute_sequence(curr_pose, args.pattern, args.cycles)
#         else:
#             print("Unknown instruction, doing a single small cross by default.")
#             goals = _plan_pattern_absolute_sequence(curr_pose, "cross", 1)

#         # === Execute segments ===
#         bar = tqdm.tqdm(total=len(goals), desc="Segments")
#         t_total = 0
#         try:
#             for goal_idx, goal_pose in enumerate(goals):
#                 # Refresh current pose right before planning (for robustness)
#                 curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=(goal_idx == 0))
#                 p0, q0 = _pose_split(curr_obs["cartesian_position"])
#                 p1, q1 = _pose_split(goal_pose)

#                 # Generate min-jerk waypoints
#                 for p_des, q_des in _generate_minjerk_cartesian(p0, q0, p1, q1):
#                     # Enforce per-tick safety limits
#                     p_curr, q_curr = _pose_split(curr_obs["cartesian_position"])
#                     p_limited, q_limited = _limit_step(p_curr, p_des, q_curr, q_des)
#                     action = _pose_pack(p_limited, q_limited)

#                     # === Send action ===
#                     env.step(action)  # absolute pose command for this tick

#                     # === Log + video ===
#                     left_img = curr_obs[f"{args.external_camera}_image"]
#                     video.append(left_img)
#                     save_rollout_data_list.append({
#                         "observation/exterior_image_1_left": image_tools.resize_with_pad(left_img, 224, 224),
#                         "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
#                         "observation/joint_position": curr_obs["joint_position"],
#                         "observation/gripper_position": curr_obs["gripper_position"],
#                         "target/pose7": action.copy(),
#                         "meta/segment_index": goal_idx,
#                     })

#                     # Sleep to keep 15 Hz
#                     t0 = time.time()
#                     elapsed = time.time() - t0
#                     if elapsed < DT:
#                         time.sleep(DT - elapsed)
#                     t_total += 1

#                     # Update current obs for next tick
#                     curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=False)

#                 bar.update(1)
#         except KeyboardInterrupt:
#             print("Interrupted by user. Stopping sweep.")
#         finally:
#             bar.close()

#         # === Save media & metrics ===
#         video = np.stack(video)
#         save_filename = "video_" + timestamp
#         ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

#         if args.save_rollout_to_hdf5:
#             save_rollout_to_hdf5(save_rollout_data_list, save_filename + ".hdf5")

#         # Simple success prompt
#         success: str | float | None = None
#         while not isinstance(success, float):
#             success = input("Did the sweep succeed? (y=100, n=0, or 0–100): ")
#             if success == "y":
#                 success = 1.0
#             elif success == "n":
#                 success = 0.0
#             else:
#                 success = float(success) / 100.0
#             if not (0.0 <= success <= 1.0):
#                 print(f"Success must be in [0,100]. Got {success*100:.1f}")

#         df = df.append({"success": success, "duration": t_total, "video_filename": save_filename}, ignore_index=True)
#         if input("Run another sweep? (y/n) ").strip().lower() != "y":
#             break
#         env.reset()

#     os.makedirs("results", exist_ok=True)
#     ts = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
#     csv_filename = os.path.join("results", f"eval_{ts}.csv")
#     df.to_csv(csv_filename)
#     print(f"Results saved to {csv_filename}")

# if __name__ == "__main__":
#     args: Args = tyro.cli(Args)
#     main(args)


# # # ruff: noqa

# # import contextlib
# # import dataclasses
# # import datetime
# # import faulthandler
# # import os
# # import signal
# # import time
# # from moviepy.editor import ImageSequenceClip
# # import numpy as np
# # from openpi_client import image_tools
# # from openpi_client import websocket_client_policy
# # import pandas as pd
# # from PIL import Image
# # from droid.robot_env import RobotEnv
# # import tqdm
# # import tyro
# # import h5py

# # faulthandler.enable()

# # # DROID data collection frequency -- we slow down execution to match this frequency
# # DROID_CONTROL_FREQUENCY = 15
# # WRIST_TORQUE_BIAS = 0.18 # NOTE: hack for RL2 Franka droid

# # import sys, select, termios, tty

# # def save_rollout_to_hdf5(rollout_data_list, filename):
# #     """
# #     Save a list of rollout data (dicts) to an HDF5 file.
# #     Each key in the dict will be a dataset; values must be numpy arrays or convertible.
# #     """
# #     with h5py.File(filename, 'w') as f:
# #         if not rollout_data_list:
# #             raise ValueError("rollout_data_list is empty")
# #         keys = rollout_data_list[0].keys()
# #         for key in keys:
# #             data = np.array([d[key] for d in rollout_data_list])
# #             # Handle string/object dtype
# #             if data.dtype.kind in {'U', 'O'}:
# #                 dt = h5py.string_dtype(encoding='utf-8')
# #                 f.create_dataset(key, data=data.astype('U'), dtype=dt)
# #             else:
# #                 f.create_dataset(key, data=data)
# #     print(f"Saved rollout data to {filename}")

# # @dataclasses.dataclass
# # class Args:
# #     # Hardware parameters
# #     left_camera_id: str = "33087938"  # e.g., "24259877"
# #     # right_camera_id: str = "33087938"  # e.g., "24514023"
# #     wrist_camera_id: str = "18482824"  # e.g., "13062452"

# #     # Policy parameters
# #     external_camera: str = (
# #         None  # which external camera should be fed to the policy, choose from ["left", "right"]
# #     )

# #     # Rollout parameters
# #     max_timesteps: int = 600
# #     # How many actions to execute from a predicted action chunk before querying policy server again
# #     # 8 is usually a good default (equals 0.5 seconds of action execution).
# #     open_loop_horizon: int = 8

# #     # Remote server parameters
# #     remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
# #     remote_port: int = (
# #         8000  # point this to the port of the policy server, default server port for openpi servers is 8000
# #     )

# #     # Aux tool
# #     save_rollout_to_hdf5: bool = False

# #     # dirty fix
# #     torque_bias: bool = False # hack for RL2 Franka droid


# # # We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# # # waiting for a new action chunk, it will raise an exception and the server connection dies.
# # # This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
# # @contextlib.contextmanager
# # def prevent_keyboard_interrupt():
# #     """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
# #     interrupted = False
# #     original_handler = signal.getsignal(signal.SIGINT)

# #     def handler(signum, frame):
# #         nonlocal interrupted
# #         interrupted = True

# #     signal.signal(signal.SIGINT, handler)
# #     try:
# #         yield
# #     finally:
# #         signal.signal(signal.SIGINT, original_handler)
# #         if interrupted:
# #             raise KeyboardInterrupt


# # def main(args: Args):
# #     # Make sure external camera is specified by user -- we only use one external camera for the policy
# #     assert (
# #         args.external_camera is not None and args.external_camera in ["left", "right"]
# #     ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

# #     # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
# #     # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
# #     env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
# #     print("Created the droid env!")

# #     df = pd.DataFrame(columns=["success", "duration", "video_filename"])

# #     while True:
# #         instruction = None
# #         while instruction is None:
# #             instruction = input("Enter instruction: ")
# #             if instruction == 'reset':
# #                 env.reset()
# #                 instruction = None

# #         # Rollout parameters
# #         actions_from_chunk_completed = 0
# #         pred_action_chunk = None

# #         # Prepare to save video of rollout
# #         timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
# #         save_rollout_data_list = []
# #         video = []
# #         bar = tqdm.tqdm(range(args.max_timesteps))
# #         print("Running rollout... press Ctrl+C to stop early.")
# #         for t_step in bar:
# #             start_time = time.time()
# #             try:

# #                 # Get the current observation
# #                 curr_obs = _extract_observation(
# #                     args,
# #                     env.get_observation(),
# #                     # Save the first observation to disk
# #                     save_to_disk=t_step == 0,
# #                 )

# #                 curr_cartesian_pose = curr_obs["cartesian_position"]
# #                 action = curr_cartesian_pose.copy()
# #                 action[0] 

# #                 video.append(curr_obs[f"{args.external_camera}_image"])
                

# #                 ###################
# #                 # Send websocket request to policy server if it's time to predict a new chunk
# #                 if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
# #                     actions_from_chunk_completed = 0

# #                     # We resize images on the robot laptop to minimize the amount of data sent to the policy server
# #                     # and improve latency.
# #                     request_data = {
# #                         "observation/exterior_image_1_left": image_tools.resize_with_pad(
# #                             curr_obs[f"{args.external_camera}_image"], 224, 224
# #                         ),
# #                         "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
# #                         "observation/joint_position": curr_obs["joint_position"],
# #                         "observation/gripper_position": curr_obs["gripper_position"],
# #                         "prompt": instruction,
# #                     }
# #                     save_rollout_data_list.append(request_data)

# #                 # Sleep to match DROID data collection frequency
# #                 elapsed_time = time.time() - start_time
# #                 if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
# #                     time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
# #             except KeyboardInterrupt:
# #                 break

# #         video = np.stack(video)
# #         save_filename = "video_" + timestamp
# #         ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

# #         ### Save all data ###
# #         if args.save_rollout_to_hdf5:
# #             # import pdb; pdb.set_trace()
# #             save_rollout_to_hdf5(save_rollout_data_list, save_filename + ".hdf5")

# #         success: str | float | None = None
# #         while not isinstance(success, float):
# #             success = input(
# #                 "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
# #             )
# #             if success == "y":
# #                 success = 1.0
# #             elif success == "n":
# #                 success = 0.0

# #             success = float(success) / 100
# #             if not (0 <= success <= 1):
# #                 print(f"Success must be a number in [0, 100] but got: {success * 100}")

# #         df = df.append(
# #             {
# #                 "success": success,
# #                 "duration": t_step,
# #                 "video_filename": save_filename,
# #             },
# #             ignore_index=True,
# #         )

# #         if input("Do one more eval? (enter y or n) ").lower() != "y":
# #             break
# #         env.reset()

# #     os.makedirs("results", exist_ok=True)
# #     timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
# #     csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
# #     df.to_csv(csv_filename)
# #     print(f"Results saved to {csv_filename}")


# # def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
# #     print("Extracting observations...")
# #     print(f"Keys in obs_dict: {list(obs_dict.keys())}")
# #     image_observations = obs_dict["image"]
# #     left_image, right_image, wrist_image = None, None, None
# #     for key in image_observations:
# #         # Note the "left" below refers to the left camera in the stereo pair.
# #         # The model is only trained on left stereo cams, so we only feed those.
# #         if args.left_camera_id in key and "left" in key:
# #             left_image = image_observations[key]
# #         # elif args.right_camera_id in key and "left" in key:
# #         #     right_image = image_observations[key]
# #         elif args.wrist_camera_id in key and "left" in key:
# #             wrist_image = image_observations[key]

# #     # Drop the alpha dimension
# #     left_image = left_image[..., :3]
# #     # right_image = right_image[..., :3]
# #     wrist_image = wrist_image[..., :3]

# #     # Convert to RGB
# #     left_image = left_image[..., ::-1]
# #     # right_image = right_image[..., ::-1]
# #     wrist_image = wrist_image[..., ::-1]

# #     # In addition to image observations, also capture the proprioceptive state
# #     robot_state = obs_dict["robot_state"]
# #     cartesian_position = np.array(robot_state["cartesian_position"])
# #     joint_position = np.array(robot_state["joint_positions"])
# #     gripper_position = np.array([robot_state["gripper_position"]])

# #     # Save the images to disk so that they can be viewed live while the robot is running
# #     # Create one combined image to make live viewing easy
# #     if save_to_disk:
# #         combined_image = np.concatenate([left_image, wrist_image], axis=1)
# #         combined_image = Image.fromarray(combined_image)
# #         combined_image.save("robot_camera_views.png")

# #     return {
# #         "left_image": left_image,
# #         # "right_image": right_image,
# #         "wrist_image": wrist_image,
# #         "cartesian_position": cartesian_position,
# #         "joint_position": joint_position,
# #         "gripper_position": gripper_position,
# #     }


# # if __name__ == "__main__":
# #     args: Args = tyro.cli(Args)
# #     main(args)
