
python /mnt/data2/droid/droid/scripts/collect_interpolated_trajectory.py \
    --traj1_file /mnt/data2/droid/droid/data/success/2025-12-24/Wed_Dec_24_17:40:54_2025/trajectory.h5 \
    --traj2_file /mnt/data2/droid/droid/data/success/2025-12-24/Wed_Dec_24_17:40:54_2025/trajectory.h5 \
    --state1_index 50 \
    --state2_index 15 \
    --num_steps 35 \
    --output_dir /mnt/data2/droid/droid/data/test/1225_3 \
    --action_space cartesian_position \
    --gripper_action_space position