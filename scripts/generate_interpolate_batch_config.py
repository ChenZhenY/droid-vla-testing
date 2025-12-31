"""
Script to generate YAML config file for batch trajectory collection.

This script:
1. Reads aggregated_instructions.json to find demos matching specified tasks
2. For each task pair, randomly samples demo and state IDs
3. Ensures action/stage matches between the two states
4. Skips first 20 and last 20 steps
5. Generates statistics and saves them in the YAML
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import h5py
import numpy as np
import yaml
import tyro


@dataclass
class Args:
    """Generate YAML config file for batch trajectory collection."""

    dataset_dir: Path
    """Path to dataset directory (e.g., /mnt/data2/droid/droid/data/success/2025-12-27)."""

    tasks: str
    """Comma-separated list of task names to generate pairs from (e.g., "task1,task2,task3")."""

    stage: int = 1
    """Stage value to match between trajectory pairs (0=PREGRASPING, 1=TRANSPORT, 2=PLACE)."""

    bidirection: bool = False
    """Whether to generate bidirectional trajectory pairs (default: False)."""

    num_demos: int = 10
    """Number of trajectory pairs to generate."""

    output_yaml: Path = Path("./data/batch_config_interpolate.yaml")
    """Output YAML config file path."""

    output_dir: Path = Path("./data/interpolation/2025-12-30")
    """Output directory for saving the collected trajectory, following DROID structure with output_dir/success/<date>/<folder_stamp>/"""

    skip_first: int = 20
    """Number of steps to skip at the beginning of trajectories."""

    skip_last: int = 20
    """Number of steps to skip at the end of trajectories."""

    step_delta: Optional[float] = 0.15
    """Step size for interpolation (default: 0.25 meters). This will disable the num_steps parameter."""

    num_steps: int = 100
    """Number of interpolation steps (default: 100)."""

    action_space: str = "cartesian_position"
    """Action space for robot control (default: cartesian_position)."""

    gripper_action_space: str = "position"
    """Gripper action space (default: position)."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""


def load_aggregated_instructions(dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Load aggregated_instructions.json from dataset directory.
    
    Returns:
        Dictionary mapping UUID to task information
    """
    instructions_file = dataset_dir / "aggregated_instructions.json"
    if not instructions_file.exists():
        raise FileNotFoundError(f"aggregated_instructions.json not found in {dataset_dir}")
    
    with open(instructions_file, 'r') as f:
        return json.load(f)


def get_task_demos(instructions: Dict[str, Dict[str, str]], task_name: str) -> List[str]:
    """
    Find all demo UUIDs that match the given task name.
    
    Args:
        instructions: Dictionary from aggregated_instructions.json
        task_name: Task name to search for (searches in language_instruction fields)
    
    Returns:
        List of UUIDs matching the task
    """
    matching_uuids = []
    task_lower = task_name.lower()
    
    for uuid, task_info in instructions.items():
        # Search in all language instruction fields
        for key, value in task_info.items():
            if isinstance(value, str) and task_lower in value.lower():
                matching_uuids.append(uuid)
                break
    
    return matching_uuids


def get_trajectory_path(dataset_dir: Path, uuid: str) -> Optional[Path]:
    """
    Find trajectory.h5 file for a given UUID.
    
    Args:
        dataset_dir: Dataset directory
        uuid: UUID from aggregated_instructions.json
    
    Returns:
        Path to trajectory.h5 file, or None if not found
    """
    # UUID format: "RL2+92f0f2ff+2025-12-24-15h-30m-01s"
    # Directory format: "Wed_Dec_24_15:30:01_2025"
    # We need to find the directory that contains this UUID
    
    # Search for metadata files with this UUID
    for metadata_file in dataset_dir.rglob(f"metadata_{uuid}.json"):
        traj_dir = metadata_file.parent
        traj_file = traj_dir / "trajectory.h5"
        if traj_file.exists():
            return traj_file
    
    return None


def read_stages(h5_filepath: Path) -> Optional[np.ndarray]:
    """
    Read action/stage array from HDF5 file.
    
    Args:
        h5_filepath: Path to trajectory.h5 file
    
    Returns:
        Array of stage values, or None if not found
    """
    try:
        with h5py.File(h5_filepath, 'r') as f:
            if "action" in f and "stage" in f["action"]:
                return np.array(f["action"]["stage"][:])
            else:
                return None
    except Exception as e:
        print(f"Error reading stages from {h5_filepath}: {e}")
        return None


def find_matching_stage_indices(
    stages: np.ndarray,
    target_stage: int,
    skip_first: int,
    skip_last: int
) -> List[int]:
    """
    Find all indices where stage matches target_stage, excluding first and last steps.
    
    Args:
        stages: Array of stage values
        target_stage: Stage value to match
        skip_first: Number of steps to skip at beginning
        skip_last: Number of steps to skip at end
    
    Returns:
        List of valid indices
    """
    if len(stages) <= skip_first + skip_last:
        return []
    
    valid_range = stages[skip_first:-skip_last if skip_last > 0 else None]
    matching_indices = np.where(valid_range == target_stage)[0]
    
    # Adjust indices to account for skipped first steps
    return (matching_indices + skip_first).tolist()


def sample_trajectory_pair(
    task1_demos: List[str],
    task2_demos: List[str],
    dataset_dir: Path,
    target_stage: int,
    skip_first: int,
    skip_last: int
) -> Optional[Tuple[str, str, int, int]]:
    """
    Sample a valid trajectory pair with matching stages.
    
    Args:
        task1_demos: List of UUIDs for task1
        task2_demos: List of UUIDs for task2
        dataset_dir: Dataset directory
        target_stage: Stage value to match
        skip_first: Number of steps to skip at beginning
        skip_last: Number of steps to skip at end
    
    Returns:
        Tuple of (traj1_uuid, traj2_uuid, state1_index, state2_index) or None if no valid pair found
    """
    # Shuffle to randomize selection
    random.shuffle(task1_demos)
    random.shuffle(task2_demos)
    
    for traj1_uuid in task1_demos:
        traj1_path = get_trajectory_path(dataset_dir, traj1_uuid)
        if traj1_path is None:
            continue
        
        stages1 = read_stages(traj1_path)
        if stages1 is None:
            continue
        
        valid_indices1 = find_matching_stage_indices(stages1, target_stage, skip_first, skip_last)
        if len(valid_indices1) == 0:
            continue
        
        # Try to find matching state in task2 demos
        for traj2_uuid in task2_demos:
            traj2_path = get_trajectory_path(dataset_dir, traj2_uuid)
            if traj2_path is None:
                continue
            
            stages2 = read_stages(traj2_path)
            if stages2 is None:
                continue
            
            valid_indices2 = find_matching_stage_indices(stages2, target_stage, skip_first, skip_last)
            if len(valid_indices2) == 0:
                continue
            
            # Sample random indices from both
            state1_index = random.choice(valid_indices1)
            state2_index = random.choice(valid_indices2)
            
            return (traj1_uuid, traj2_uuid, state1_index, state2_index)
    
    return None


def generate_batch_config(
    dataset_dir: Path,
    output_dir: Path,
    task_list: List[str],
    target_stage: int,
    num_demos: int,
    skip_first: int,
    skip_last: int,
    num_steps: int,
    action_space: str,
    gripper_action_space: str,
    step_delta: Optional[float] = None,
    bidirection: bool = False,
) -> Dict[str, Any]:
    """
    Generate batch config dictionary.
    
    Returns:
        Dictionary containing config and statistics
    """
    # Load instructions
    instructions = load_aggregated_instructions(dataset_dir)

    print(f"Found {len(instructions)} instructions, {instructions}")
    
    # Get demos for each task
    task_demos = {}
    for task in task_list:
        demos = get_task_demos(instructions, task)
        task_demos[task] = demos
        print(f"Task '{task}': found {len(demos)} demos")
    
    # Generate all task pairs
    task_pairs = []
    for i, task1 in enumerate(task_list):
        for task2 in task_list[i+1:]:
            task_pairs.append((task1, task2))
    
    if len(task_pairs) == 0:
        raise ValueError("Need at least 2 tasks to generate pairs")
    
    print(f"Generating {num_demos} trajectory pairs from {len(task_pairs)} task pairs")
    
    # Sample trajectory pairs
    trajectories = []
    stats = {
        "total_pairs_requested": num_demos,
        "total_pairs_generated": 0,
        "pairs_per_task_pair": {},
        "failed_samples": 0,
    }
    
    for task1, task2 in task_pairs:
        task1_demos = task_demos[task1]
        task2_demos = task_demos[task2]
        
        if len(task1_demos) == 0:
            print(f"Warning: No demos found for task '{task1}'")
            continue
        if len(task2_demos) == 0:
            print(f"Warning: No demos found for task '{task2}'")
            continue
        
        pairs_for_this_task_pair = 0
        
        # Sample pairs for this task combination
        for _ in range(num_demos):
            result = sample_trajectory_pair(
                task1_demos,
                task2_demos,
                dataset_dir,
                target_stage,
                skip_first,
                skip_last,
            )
            
            if result is None:
                stats["failed_samples"] += 1
                continue
            
            traj1_uuid, traj2_uuid, state1_index, state2_index = result
            
            traj1_path = get_trajectory_path(dataset_dir, traj1_uuid)
            traj2_path = get_trajectory_path(dataset_dir, traj2_uuid)
            
            if traj1_path is None or traj2_path is None:
                stats["failed_samples"] += 1
                continue
            
            pair_name = f"{task1}_to_{task2}_{pairs_for_this_task_pair + 1}"
            trajectories.append({
                "name": pair_name,
                "traj1_file": str(traj1_path),
                "traj2_file": str(traj2_path),
                "state1_index": int(state1_index),
                "state2_index": int(state2_index),
                "target_stage": int(target_stage),
            })
            
            pairs_for_this_task_pair += 1
            stats["total_pairs_generated"] += 1
        
        stats["pairs_per_task_pair"][f"{task1}_to_{task2}"] = pairs_for_this_task_pair
        
        # Generate reverse pairs (task2->task1) if bidirection is True
        if bidirection:
            pairs_for_reverse = 0
            
            # Sample pairs for task2->task1
            for _ in range(num_demos):
                result = sample_trajectory_pair(
                    task2_demos,  # Swap: task2 becomes task1
                    task1_demos,  # Swap: task1 becomes task2
                    dataset_dir,
                    target_stage,
                    skip_first,
                    skip_last,
                )
                
                if result is None:
                    stats["failed_samples"] += 1
                    continue
                
                traj2_uuid, traj1_uuid, state2_index, state1_index = result
                
                traj2_path = get_trajectory_path(dataset_dir, traj2_uuid)
                traj1_path = get_trajectory_path(dataset_dir, traj1_uuid)
                
                if traj2_path is None or traj1_path is None:
                    stats["failed_samples"] += 1
                    continue
                
                pair_name = f"{task2}_to_{task1}_{pairs_for_reverse + 1}"
                trajectories.append({
                    "name": pair_name,
                    "traj1_file": str(traj2_path),  # traj1_file from task2
                    "traj2_file": str(traj1_path),  # traj2_file from task1
                    "state1_index": int(state2_index),  # state1_index from task2
                    "state2_index": int(state1_index),  # state2_index from task1
                    "target_stage": int(target_stage),
                })
                
                pairs_for_reverse += 1
                stats["total_pairs_generated"] += 1
            
            stats["pairs_per_task_pair"][f"{task2}_to_{task1}"] = pairs_for_reverse
    
    # Build config
    # output_dir should be the parent of the "success" directory
    # If dataset_dir is /path/to/data/success/2025-12-27, output_dir should be /path/to/data
    # The collect script creates: output_dir/success/<date>/<folder_stamp>/
    # if dataset_dir.parent.name == "success":
    #     output_dir = dataset_dir.parent.parent  # Go up two levels: date -> success -> data
    # else:
    #     output_dir = dataset_dir.parent  # Fallback
    
    config = {
        "output_dir": str(output_dir),
        "num_steps": num_steps,
        "step_delta": step_delta,
        "action_space": action_space,
        "gripper_action_space": gripper_action_space,
        "trajectories": trajectories,
        "statistics": {
            "dataset_dir": str(dataset_dir),
            "tasks": task_list,
            "target_stage": target_stage,
            "skip_first": skip_first,
            "skip_last": skip_last,
            **stats,
        },
    }
    
    return config


def main():
    args = tyro.cli(Args)
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Parse task list
    task_list = [task.strip() for task in args.tasks.split(",")]
    if len(task_list) < 2:
        raise ValueError("Need at least 2 tasks to generate pairs")
    
    print(f"Generating batch config for tasks: {task_list}")
    print(f"Target stage: {args.stage}")
    print(f"Number of demos per task pair: {args.num_demos}")
    if args.bidirection:
        print(f"Bidirectional mode enabled: will generate both task1->task2 and task2->task1 pairs")
    
    # Generate config
    config = generate_batch_config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        task_list=task_list,
        target_stage=args.stage,
        num_demos=args.num_demos,
        skip_first=args.skip_first,
        skip_last=args.skip_last,
        num_steps=args.num_steps,
        action_space=args.action_space,
        gripper_action_space=args.gripper_action_space,
        step_delta=args.step_delta,
        bidirection=args.bidirection,
    )
    
    # Save to YAML
    with open(args.output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"\nBatch config saved to: {args.output_yaml}")
    print(f"Total trajectories generated: {len(config['trajectories'])}")
    print(f"Statistics:")
    print(f"  - Total pairs requested: {config['statistics']['total_pairs_requested']}")
    print(f"  - Total pairs generated: {config['statistics']['total_pairs_generated']}")
    print(f"  - Failed samples: {config['statistics']['failed_samples']}")
    print(f"  - Pairs per task combination:")
    for task_pair, count in config['statistics']['pairs_per_task_pair'].items():
        print(f"    - {task_pair}: {count}")


if __name__ == "__main__":
    main()

