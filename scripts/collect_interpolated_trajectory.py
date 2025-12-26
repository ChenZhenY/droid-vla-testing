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

from droid.robot_env import RobotEnv
from droid.trajectory_utils.trajectory_reader import TrajectoryReader
from droid.trajectory_utils.misc import collect_trajectory
from droid.misc.transformations import euler_to_quat, quat_to_euler


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


def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectory by interpolating between two states from two HDF5 files"
    )
    parser.add_argument(
        "--traj1_file",
        type=str,
        required=True,
        help="Path to first trajectory HDF5 file",
    )
    parser.add_argument(
        "--traj2_file",
        type=str,
        required=True,
        help="Path to second trajectory HDF5 file",
    )
    parser.add_argument(
        "--state1_index",
        type=int,
        required=True,
        help="Index of state to read from first trajectory (0-based)",
    )
    parser.add_argument(
        "--state2_index",
        type=int,
        required=True,
        help="Index of state to read from second trajectory (0-based)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of interpolation steps (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for saving the collected trajectory",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="cartesian_position",
        choices=["cartesian_position", "cartesian_velocity"],
        help="Action space for robot control (default: cartesian_position)",
    )
    parser.add_argument(
        "--gripper_action_space",
        type=str,
        default="position",
        choices=["position", "velocity"],
        help="Gripper action space (default: position)",
    )

    args = parser.parse_args()

    # Read states from trajectory files
    print(f"Reading state {args.state1_index} from {args.traj1_file}...")
    state1 = read_state_from_trajectory(args.traj1_file, args.state1_index)

    print(f"Reading state {args.state2_index} from {args.traj2_file}...")
    state2 = read_state_from_trajectory(args.traj2_file, args.state2_index)

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
    metadata = {
        "traj1_file": args.traj1_file,
        "traj2_file": args.traj2_file,
        "state1_index": args.state1_index,
        "state2_index": args.state2_index,
        "num_steps": args.num_steps,
        "interpolation_type": "linear_position_slerp_orientation",
    }

    # Collect trajectory
    print(f"Collecting trajectory with {args.num_steps} steps...")
    print(f"Output will be saved to: {args.output_dir}")

    # Ensure output directory exists
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    recording_folderpath = os.path.join(args.output_dir, "recordings")
    save_filepath = os.path.join(args.output_dir, "trajectory.h5")

    # move the robot to initial position before collecting
    print("Moving robot to initial position...")
    init_action = np.concatenate([cartesian1, [gripper1]])
    env.step(init_action, action_space=args.action_space, gripper_action_space=args.gripper_action_space, blocking=True)

    time.sleep(1.0)  # wait for a moment
    print(f"Starting Trajectory Collection...")
    # NOTE: save_images=True will bug out
    controller_info = collect_trajectory(
        env=env,
        policy=policy,
        horizon=args.num_steps,
        save_filepath=save_filepath,
        metadata=metadata,
        recording_folderpath=recording_folderpath,
        reset_robot=False,
    )

    print("Trajectory collection complete!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
