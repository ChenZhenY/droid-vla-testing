from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import replay_trajectory

trajectory_folderpath = "/home/sasha/DROID/data/success/2023-02-16/Thu_Feb_16_16:27:00_2023"
trajectory_folderpath = "/mnt/data2/droid/droid/data/success/2025-12-16/Tue_Dec_16_18:10:45_2025"
trajectory_folderpath = "/mnt/data2/droid/droid/data/success/2025-12-24/Wed_Dec_24_17:05:08_2025"
# action_space = "joint_position"
action_space = "joint_velocity"

# Make the robot env
env = RobotEnv(action_space=action_space)

# Replay Trajectory #
h5_filepath = trajectory_folderpath + "/trajectory.h5"
replay_trajectory(env, filepath=h5_filepath)
