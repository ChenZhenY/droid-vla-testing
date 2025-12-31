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
import yaml
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import tyro

from droid.robot_env import RobotEnv
from droid.trajectory_utils.trajectory_reader import TrajectoryReader
from droid.trajectory_utils.misc import collect_trajectory
from droid.misc.transformations import euler_to_quat, quat_to_euler

@dataclass
class Args:
    """Collect trajectory by interpolating between two states from two HDF5 files."""

    config_file: Optional[Path] = None
    """Path to YAML config file for batch trajectory collection. If provided, other args are ignored."""

    traj1_file: Optional[Path] = None
    """Path to first trajectory HDF5 file."""

    traj2_file: Optional[Path] = None
    """Path to second trajectory HDF5 file."""

    state1_index: Optional[int] = None
    """Index of state to read from first trajectory (0-based)."""

    state2_index: Optional[int] = None
    """Index of state to read from second trajectory (0-based)."""
    
    output_dir: Optional[Path] = None
    """Path to output directory for saving the collected trajectory, following DROID structure with output_dir/success/<date>/<folder_stamp>/"""

    step_delta: Optional[float] = 0.25 # m
    """Step size for interpolation (default: 0.25 meters). This will disable the num_steps parameter."""

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
        print("\033[91mFailed to reach target state within threshold.\033[0m")
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

def collect_single_trajectory(
    env: RobotEnv,
    traj1_file: Path,
    traj2_file: Path,
    state1_index: int,
    state2_index: int,
    output_dir: Path,
    num_steps: int,
    action_space: str,
    gripper_action_space: str,
    trajectory_name: Optional[str] = None,
    step_delta: Optional[float] = None,
) -> Path:
    """
    Collect a single interpolated trajectory between two states.
    
    Args:
        traj1_file: Path to first trajectory HDF5 file
        traj2_file: Path to second trajectory HDF5 file
        state1_index: Index of state to read from first trajectory (0-based)
        state2_index: Index of state to read from second trajectory (0-based)
        output_dir: Base output directory for saving trajectories
        num_steps: Number of interpolation steps (used if step_delta is None)
        action_space: Action space for robot control
        gripper_action_space: Gripper action space
        trajectory_name: Optional name for the trajectory (for logging)
        step_delta: Step size for interpolation in meters. If provided, num_steps will be calculated from the distance between states.
    
    Returns:
        Path to the saved trajectory directory
    """
    name_prefix = f"[{trajectory_name}] " if trajectory_name else ""

    # get timestamp
    now = datetime.now()
    new_date = now.strftime("%Y-%m-%d")
    new_timestamp = now.strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
    new_folder_stamp = now.strftime("%a_%b_%d_%H:%M:%S_%Y")

    # Read states from trajectory files
    state1 = read_state_from_trajectory(traj1_file, state1_index)
    state2 = read_state_from_trajectory(traj2_file, state2_index)

    # Extract cartesian positions and gripper positions
    cartesian1 = np.array(state1["observation"]["robot_state"]["cartesian_position"])
    gripper1 = state1["observation"]["robot_state"]["gripper_position"]
    
    cartesian2 = np.array(state2["observation"]["robot_state"]["cartesian_position"])
    gripper2 = state2["observation"]["robot_state"]["gripper_position"]

    # move the robot to initial position before collecting
    print(f"{name_prefix}Moving robot to initial position...")
    current_state, _ = env.get_state()
    success = go_to_state(env, current_state, cartesian1, gripper1)
    if not success:
        print(f"{name_prefix}Failed to move robot to initial position.")
        return None
    time.sleep(0.5)  # wait for a moment
    
    # Calculate num_steps from step_delta if provided
    if step_delta is not None:
        distance = np.linalg.norm(cartesian2[:3] - cartesian1[:3])
        num_steps = max(1, int(np.ceil(distance / step_delta)))
        print(f"{name_prefix}Calculated {num_steps} steps from distance {distance:.3f}m and step_delta {step_delta}m")
    else:
        print(f"{name_prefix}Collecting trajectory with {num_steps} steps...")

    # Save updated metadata JSON
    traj1_folder = os.path.dirname(traj1_file)
    json_files1 = sorted(Path(traj1_folder).expanduser().rglob("*.json"))
    meta = json.load(open(json_files1[0], "r"))
    start_task = meta.get("current_task", "unknown task")

    # create the output directory following DROID if it does not exist
    trajectory_output_dir = os.path.join(output_dir, "success", new_date, new_folder_stamp)
    if trajectory_output_dir and not os.path.exists(trajectory_output_dir):
        os.makedirs(trajectory_output_dir)
    recording_folderpath = os.path.join(trajectory_output_dir, "recordings")
    save_filepath = os.path.join(trajectory_output_dir, "trajectory.h5")

    traj2_folder = os.path.dirname(traj2_file)
    json_files2 = sorted(Path(traj2_folder).expanduser().rglob("*.json"))
    meta2 = json.load(open(json_files2[0], "r"))
    target_task = meta2.get("current_task", "unknown task")
    save_metadata_json_with_current_time(
        metadata_template=meta,
        out_dir=trajectory_output_dir,
        start_task=start_task,
        start_step=state1_index,
        current_task=target_task,
        current_step=state2_index,
        now=now,
    )

    # Create interpolation policy
    policy = InterpolationPolicy(
        start_cartesian=cartesian1,
        end_cartesian=cartesian2,
        start_gripper=gripper1,
        end_gripper=gripper2,
        num_steps=num_steps,
    )

    print(f"{name_prefix}Starting Trajectory Collection...")
    # NOTE: save_images=True will bug out
    controller_info = collect_trajectory(
        env=env,
        policy=policy,
        horizon=num_steps,
        save_filepath=save_filepath,
        recording_folderpath=recording_folderpath,
        reset_robot=False,
    )

    print(f"{name_prefix}Trajectory collection complete!")
    print(f"{name_prefix}Saved to: {trajectory_output_dir}")
    
    return Path(trajectory_output_dir)

def load_batch_config(config_file: Path) -> Dict[str, Any]:
    """
    Load and validate batch trajectory collection config from YAML file.
    
    Args:
        config_file: Path to YAML config file
    
    Returns:
        Dictionary containing validated config data
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config structure is invalid
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a dictionary")
    
    # Validate required fields
    if "output_dir" not in config:
        raise ValueError("Config must contain 'output_dir' field")
    
    if "trajectories" not in config:
        raise ValueError("Config must contain 'trajectories' list")
    
    if not isinstance(config["trajectories"], list):
        raise ValueError("'trajectories' must be a list")
    
    if len(config["trajectories"]) == 0:
        raise ValueError("'trajectories' list cannot be empty")
    
    # Validate each trajectory entry
    required_traj_fields = ["traj1_file", "traj2_file", "state1_index", "state2_index"]
    for i, traj in enumerate(config["trajectories"]):
        if not isinstance(traj, dict):
            raise ValueError(f"Trajectory {i} must be a dictionary")
        
        for field in required_traj_fields:
            if field not in traj:
                raise ValueError(f"Trajectory {i} missing required field: {field}")
        
        # Convert paths to Path objects
        traj["traj1_file"] = Path(traj["traj1_file"])
        traj["traj2_file"] = Path(traj["traj2_file"])
        
        # Validate files exist
        if not traj["traj1_file"].exists():
            raise FileNotFoundError(f"Trajectory {i}: traj1_file not found: {traj['traj1_file']}")
        if not traj["traj2_file"].exists():
            raise FileNotFoundError(f"Trajectory {i}: traj2_file not found: {traj['traj2_file']}")
    
    # Set defaults for optional fields
    if "num_steps" not in config:
        config["num_steps"] = 100
    if "step_delta" not in config:
        config["step_delta"] = None
    if "action_space" not in config:
        config["action_space"] = "cartesian_position"
    if "gripper_action_space" not in config:
        config["gripper_action_space"] = "position"
    
    # Convert output_dir to Path
    config["output_dir"] = Path(config["output_dir"])
    
    return config

def collect_batch_trajectories(env: RobotEnv, config: Dict[str, Any]) -> None:
    """
    Collect multiple trajectories from a batch config.
    
    Args:
        config: Validated batch config dictionary
    """
    trajectories = config["trajectories"]
    output_dir = config["output_dir"]
    global_num_steps = config["num_steps"]
    global_step_delta = config.get("step_delta")
    global_action_space = config["action_space"]
    global_gripper_action_space = config["gripper_action_space"]
    
    print(f"Found {len(trajectories)} trajectories to collect")
    print(f"Output directory: {output_dir}")
    step_delta_str = f"step_delta={global_step_delta}m" if global_step_delta is not None else "step_delta=None"
    print(f"Global settings: num_steps={global_num_steps}, {step_delta_str}, action_space={global_action_space}, gripper_action_space={global_gripper_action_space}")
    
    successful = []
    failed = []
    
    for idx, traj_config in enumerate(trajectories, 1):
        traj_name = traj_config.get("name", f"trajectory_{idx}")
        print(f"\n{'='*60}")
        print(f"Collecting trajectory {idx}/{len(trajectories)}: {traj_name}")
        print(f"{'='*60}")
        
        num_steps = traj_config.get("num_steps", global_num_steps)
        step_delta = traj_config.get("step_delta", global_step_delta)
        action_space = traj_config.get("action_space", global_action_space)
        gripper_action_space = traj_config.get("gripper_action_space", global_gripper_action_space)
        
        output_path = collect_single_trajectory(
            env=env,
            traj1_file=traj_config["traj1_file"],
            traj2_file=traj_config["traj2_file"],
            state1_index=traj_config["state1_index"],
            state2_index=traj_config["state2_index"],
            output_dir=output_dir,
            num_steps=num_steps,
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            trajectory_name=traj_name,
            step_delta=step_delta,
        )
        if output_path is None:
            failed.append((traj_name, "Failed to move robot to initial position."))
            print(f"✗ Failed to collect {traj_name}")
            continue
        successful.append((traj_name, output_path))
        print(f"✓ Successfully collected: {traj_name}")

        time.sleep(0.5)  # wait for a moment

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful trajectories:")
        for name, path in successful:
            print(f"  ✓ {name} -> {path}")
    
    if failed:
        print(f"\nFailed trajectories:")
        for name, error in failed:
            print(f"  ✗ {name}: {error}")
    
    print(f"\nAll trajectories saved to: {output_dir}")

def main():
    args = tyro.cli(Args)

    # Create robot environment
    print(f"Initializing robot environment...")
    env = RobotEnv(
        action_space=args.action_space,
        gripper_action_space=args.gripper_action_space,
    )
    env.reset()
    time.sleep(1.0)

    if args.config_file is not None:
        # Batch mode: load config and collect multiple trajectories
        print(f"Loading batch config from: {args.config_file}")
        config = load_batch_config(args.config_file)
        collect_batch_trajectories(env=env, config=config)
    else:
        # Single trajectory mode: use CLI args (backward compatible)
        required_fields = {
            "traj1_file": args.traj1_file,
            "traj2_file": args.traj2_file,
            "state1_index": args.state1_index,
            "state2_index": args.state2_index,
            "output_dir": args.output_dir,
        }
        
        missing = [field for field, value in required_fields.items() if value is None]
        if missing:
            missing_args = ', '.join(f'--{field.replace("_", "-")}' for field in missing)
            raise ValueError(
                f"When not using --config-file, all of the following must be provided: {missing_args}"
            )
        
        collect_single_trajectory(
            env=env,
            traj1_file=args.traj1_file,
            traj2_file=args.traj2_file,
            state1_index=args.state1_index,
            state2_index=args.state2_index,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            action_space=args.action_space,
            gripper_action_space=args.gripper_action_space,
            step_delta=args.step_delta,
        )


if __name__ == "__main__":
    main()
