"""
Script to collect a trajectory by interpolating between two states from two HDF5 trajectory files.

This script:
1. Reads specific states from two trajectory HDF5 files
2. Extracts cartesian space coordinates from those states
3. Interpolates between the two coordinates to generate a trajectory
4. Wraps the trajectory as a policy and commands the robot accordingly
5. Saves the trajectory in the same HDF5 format

Things to test:
1. Replay with Cartesian position 6D pose
2. Interpoaltion output (dry run)
3. Test full pipeline and whether the controller works
"""

import argparse
import os
import numpy as np
import time
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
import tyro

from droid.robot_env import RobotEnv
from droid.trajectory_utils.trajectory_reader import TrajectoryReader
from droid.trajectory_utils.misc import collect_trajectory
from droid.misc.transformations import euler_to_quat, quat_to_euler

@dataclass
class Args:
    """Collect trajectory by interpolating between two states from two HDF5 files."""

    traj1_file: Path
    """Path to first trajectory HDF5 file."""

    traj2_file: Path
    """Path to second trajectory HDF5 file."""

    state1_index: int
    """Index of state to read from first trajectory (0-based)."""

    state2_index: int
    """Index of state to read from second trajectory (0-based)."""
    
    output_dir: Path
    """Path to output directory for saving the collected trajectory, following DROID structure with output_dir/success/<date>/<folder_stamp>/"""

    num_steps: int = 100
    """Number of interpolation steps (default: 100)."""

    action_space: str = "cartesian_position"
    # action_space: Literal["cartesian_position", "cartesian_velocity"] = "cartesian_position"
    """Action space for robot control (default: cartesian_position)."""

    # gripper_action_space: Literal["position", "velocity"] = "position"
    gripper_action_space: str = "position"
    """Gripper action space (default: position)."""

class InterpolationPolicy:
    """Policy that outputs interpolated cartesian positions between two states."""

    def __init__(self, start_cartesian, end_cartesian, start_gripper, end_gripper, num_steps):
        """
        Args:
            start_cartesian: Starting cartesian position [x, y, z, roll, pitch, yaw]
            end_cartesian: Ending cartesian position [x, y, z, roll, pitch, yaw]
            start_gripper: Starting gripper position
            end_gripper: Ending gripper position
            num_steps: Number of interpolation steps
        """
        self.start_cartesian = np.array(start_cartesian)
        self.end_cartesian = np.array(end_cartesian)
        self.start_gripper = start_gripper
        self.end_gripper = end_gripper
        self.num_steps = num_steps
        self.step_count = 0

        # Pre-compute interpolated trajectory
        self.trajectory = self._generate_trajectory()

    def _generate_trajectory(self):
        """Generate interpolated trajectory between start and end states."""
        trajectory = []

        # Extract position and orientation
        start_pos = self.start_cartesian[:3]
        start_euler = self.start_cartesian[3:6]
        end_pos = self.end_cartesian[:3]
        end_euler = self.end_cartesian[3:6]

        # Convert to quaternions for SLERP
        start_quat = euler_to_quat(start_euler)
        end_quat = euler_to_quat(end_euler)

        for i in range(self.num_steps):
            alpha = i / (self.num_steps - 1) if self.num_steps > 1 else 0.0

            # Linear interpolation for position
            pos = start_pos + alpha * (end_pos - start_pos)

            # SLERP for orientation
            quat = self._slerp(start_quat, end_quat, alpha)
            euler = quat_to_euler(quat)

            # Linear interpolation for gripper
            gripper = self.start_gripper + alpha * (self.end_gripper - self.start_gripper)

            # Combine into cartesian position [x, y, z, roll, pitch, yaw]
            cartesian = np.concatenate([pos, euler])
            trajectory.append((cartesian, gripper))

        return trajectory

    def _slerp(self, q0, q1, t):
        """Spherical linear interpolation between two quaternions."""
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)

        dot = np.dot(q0, q1)

        # If dot product is negative, negate one quaternion to take shorter path
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        # If quaternions are very close, use linear interpolation
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            q = q0 + t * (q1 - q0)
            return q / np.linalg.norm(q)

        # SLERP
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        return s0 * q0 + s1 * q1

    def forward(self, observation):
        """Return the next action in the interpolated trajectory."""
        if self.step_count >= len(self.trajectory):
            # Return the last action if we've exceeded the trajectory
            cartesian, gripper = self.trajectory[-1]
        else:
            cartesian, gripper = self.trajectory[self.step_count]

        self.step_count += 1

        # Return action as [x, y, z, roll, pitch, yaw, gripper]
        action = np.concatenate([cartesian, [gripper]])
        return action

    def reset(self):
        """Reset the policy to the beginning of the trajectory."""
        self.step_count = 0


def read_state_from_trajectory(filepath, state_index):
    """
    Read a specific state from a trajectory HDF5 file.

    Args:
        filepath: Path to the HDF5 trajectory file
        state_index: Index of the state to read (0-based)

    Returns:
        Dictionary containing the state information
    """
    traj_reader = TrajectoryReader(filepath, read_images=False)
    length = traj_reader.length()

    if state_index < 0 or state_index >= length:
        raise ValueError(f"State index {state_index} is out of range [0, {length-1}]")

    timestep = traj_reader.read_timestep(index=state_index)
    traj_reader.close()

    return timestep

def go_to_state(env, current_state, target_cartesian, target_gripper,
                robot_action_space="cartesian_position", gripper_action_space="position",
                delta_threshold=0.05):
    """
    Move the robot to a target cartesian position and gripper position.

    Args:
        env: Robot environment
        current_state: Current robot state dictionary
        target_cartesian: Target cartesian position [x, y, z, roll, pitch, yaw]
        target_gripper: Target gripper position
        delta_threshold: Threshold for considering the robot to have reached the target
    Returns:
        True when the target is reached
        False if max iterations exceeded or out of ranges
    """

    delta_position = np.linalg.norm(np.array(target_cartesian[:3]) - np.array(current_state["cartesian_position"][:3]))
    steps_needed = int(delta_position / delta_threshold) + 1
    print(f"Moving to target state over approximately {steps_needed} steps.")

    policy = InterpolationPolicy(
        start_cartesian=current_state["cartesian_position"],
        end_cartesian=target_cartesian,
        start_gripper=current_state["gripper_position"],
        end_gripper=target_gripper,
        num_steps=steps_needed,
    )

    for _ in range(steps_needed):
        action = policy.forward(None)
        env.update_robot(action, action_space=robot_action_space,
                         gripper_action_space=gripper_action_space)
        time.sleep(0.2)  # small delay to allow robot to move

    env.step(np.concatenate([target_cartesian, [target_gripper]]))

    current_state, _ = env.get_state()
    final_delta = np.linalg.norm(np.array(target_cartesian[:3]) - np.array(current_state["cartesian_position"][:3]))
    if final_delta < delta_threshold:
        return True
    else:
        print("Failed to reach target state within threshold.")
        return False

def save_metadata_json_with_current_time(
    metadata_template: Dict[str, Any],
    out_dir: str,
    start_task: str,
    start_step: int,
    current_task: str = None,
    current_step: int = None,
    *,
    now: Optional[datetime] = None,
    indent: int = 2,
) -> Path:
    """
    Create and save a metadata JSON named `metadata_{uuid}.json`, substituting the time-related fields
    to "now" while leaving all other content unchanged.

    Updates:
      - "uuid": replaces the trailing timestamp segment (after the last '+') with current timestamp
      - "date": "YYYY-MM-DD"
      - "timestamp": "YYYY-MM-DD-18h-46m-30s" style
    Also updates any string value containing the old date or old "Tue_Dec_23_18:46:30_2025" style
    directory segment to the corresponding "now" version, so paths remain consistent.

    Args:
        metadata_template: original metadata dict (will not be modified in-place)
        out_dir: directory to write the JSON into
        current_task: description of the current task
        now: optional datetime (for testing); defaults to datetime.now()
        indent: json indent

    Returns:
        Path to the saved file.
    """
    assert current_task is not None, "current_task must be provided"

    if now is None:
        now = datetime.now()

    # Desired formats:
    new_date = now.strftime("%Y-%m-%d")
    new_timestamp = now.strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
    new_folder_stamp = now.strftime("%a_%b_%d_%H:%M:%S_%Y")  # e.g., Tue_Dec_23_18:46:30_2025

    md = deepcopy(metadata_template)

    # Extract old values (if present) so we can keep paths consistent.
    old_date = md.get("date")
    old_timestamp = md.get("timestamp")

    # If uuid exists, replace its trailing timestamp segment with the new timestamp.
    old_uuid = md.get("uuid")
    if isinstance(old_uuid, str) and "+" in old_uuid:
        parts = old_uuid.split("+")
        # Convention in your example: last segment is the timestamp string
        parts[-1] = new_timestamp
        new_uuid = "+".join(parts)
    elif isinstance(old_uuid, str) and old_uuid:
        # Fallback: keep uuid prefix and append timestamp
        new_uuid = f"{old_uuid}+{new_timestamp}"
    else:
        raise ValueError("metadata_template must contain a non-empty string field 'uuid'.")

    md["uuid"] = new_uuid
    md["date"] = new_date
    md["timestamp"] = new_timestamp
    md["current_task"] = current_task
    md["current_step"] = current_step
    md["start_task"] = start_task
    md["start_step"] = start_step

    # Determine old folder stamp (from hdf5_path if possible) to rewrite paths robustly.
    # Example: success/2025-12-23/Tue_Dec_23_18:46:30_2025/trajectory.h5
    old_folder_stamp = None
    hdf5_path = md.get("hdf5_path")
    if isinstance(hdf5_path, str):
        chunks = hdf5_path.split("/")
        if len(chunks) >= 3:
            # chunks[1] is date folder, chunks[2] is folder stamp in your structure
            old_folder_stamp = chunks[2]

    def _rewrite_value(v: Any) -> Any:
        if not isinstance(v, str):
            return v

        s = v
        if isinstance(old_date, str) and old_date:
            s = s.replace(old_date, new_date)
        if isinstance(old_folder_stamp, str) and old_folder_stamp:
            s = s.replace(old_folder_stamp, new_folder_stamp)

        return s

    # Rewrite all string fields (including paths) to reflect the updated date/folder stamp.
    for k, v in list(md.items()):
        md[k] = _rewrite_value(v)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"metadata_{md['uuid']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(md, f, ensure_ascii=False, indent=indent)

    return out_path

def main():
    args = tyro.cli(Args)

    # Collect trajectory
    print(f"Collecting trajectory with {args.num_steps} steps...")
    print(f"Output will be saved to: {args.output_dir}")

    # create the output directory following DROID if it does not exist
    now = datetime.now()
    new_date = now.strftime("%Y-%m-%d")
    new_timestamp = now.strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
    new_folder_stamp = now.strftime("%a_%b_%d_%H:%M:%S_%Y")
    output_dir = os.path.join(args.output_dir, "success", new_date, new_folder_stamp)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    recording_folderpath = os.path.join(output_dir, "recordings")
    save_filepath = os.path.join(output_dir, "trajectory.h5")

    # Read states from trajectory files
    print(f"Reading state {args.state1_index} from {args.traj1_file}...")
    state1 = read_state_from_trajectory(args.traj1_file, args.state1_index)

    print(f"Reading state {args.state2_index} from {args.traj2_file}...")
    state2 = read_state_from_trajectory(args.traj2_file, args.state2_index)

    # Save updated metadata JSON
    traj1_folder = os.path.dirname(args.traj1_file)
    json_files1 = sorted(Path(traj1_folder).expanduser().rglob("*.json"))
    meta = json.load(open(json_files1[0], "r"))
    start_task = meta.get("current_task", "unknown task")

    traj2_folder = os.path.dirname(args.traj2_file)
    json_files2 = sorted(Path(traj2_folder).expanduser().rglob("*.json"))
    meta2 = json.load(open(json_files2[0], "r"))
    target_task = meta2.get("current_task", "unknown task")
    save_metadata_json_with_current_time(
        metadata_template=meta,
        out_dir=output_dir,
        start_task=start_task,
        start_step=args.state1_index,
        current_task=target_task,
        current_step=args.state2_index,
        now=now,
    )

    # Extract cartesian positions and gripper positions
    cartesian1 = np.array(state1["observation"]["robot_state"]["cartesian_position"])
    gripper1 = state1["observation"]["robot_state"]["gripper_position"]

    cartesian2 = np.array(state2["observation"]["robot_state"]["cartesian_position"])
    gripper2 = state2["observation"]["robot_state"]["gripper_position"]

    print(f"State 1 - Cartesian: {cartesian1}, Gripper: {gripper1}")
    print(f"State 2 - Cartesian: {cartesian2}, Gripper: {gripper2}")

    # Create interpolation policy
    policy = InterpolationPolicy(
        start_cartesian=cartesian1,
        end_cartesian=cartesian2,
        start_gripper=gripper1,
        end_gripper=gripper2,
        num_steps=args.num_steps,
    )

    # Create robot environment
    print("Initializing robot environment...")
    env = RobotEnv(
        action_space=args.action_space,
        gripper_action_space=args.gripper_action_space,
    )

    # Prepare metadata
    # metadata = {
    #     "traj1_file": args.traj1_file,
    #     "traj2_file": args.traj2_file,
    #     "state1_index": args.state1_index,
    #     "state2_index": args.state2_index,
    #     "num_steps": args.num_steps,
    #     "interpolation_type": "linear_position_slerp_orientation",
    # }

    # move the robot to initial position before collecting
    print("Moving robot to initial position...")
    current_state, _ = env.get_state()
    go_to_state(env, current_state, cartesian1, gripper1)

    time.sleep(1.0)  # wait for a moment
    print(f"Starting Trajectory Collection...")
    # NOTE: save_images=True will bug out
    controller_info = collect_trajectory(
        env=env,
        policy=policy,
        horizon=args.num_steps,
        save_filepath=save_filepath,
        recording_folderpath=recording_folderpath,
        reset_robot=False,
    )

    print("Trajectory collection complete!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
