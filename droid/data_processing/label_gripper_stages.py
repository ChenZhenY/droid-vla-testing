"""
Script to label motion stages in DROID HDF5 trajectory files based on gripper position.

The script:
1. Reads all HDF5 files recursively from a given directory
2. Extracts action/gripper_position data (0=open, 1=close)
3. Labels motion stages: pregrasping, transport, place
4. Filters noisy/spiky signals
5. Saves stage labels back to HDF5 files
6. Generates visualization plots

Usage:
    uv run examples/droid/label_gripper_stages.py --data_dir /path/to/data
"""

from enum import IntEnum
from pathlib import Path
from typing import Optional, List, Tuple
# from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro
from tqdm import tqdm


class Stage(IntEnum):
    """Motion stage enum for gripper-based labeling."""
    PREGRASPING = 0
    TRANSPORT = 1
    PLACE = 2
    UNKNOWN = 3


def find_hdf5_files(data_dir: Path) -> List[Path]:
    """Recursively find all HDF5 files in the given directory."""
    h5_files = list(data_dir.glob("**/*.h5")) + list(data_dir.glob("**/*.hdf5"))
    return sorted(h5_files)


def read_gripper_position(h5_file: h5py.File) -> Optional[np.ndarray]:
    """
    Extract gripper_position from action/gripper_position dataset.
    
    Returns:
        numpy array of gripper positions, or None if not found
    """
    try:
        # Try direct path first
        if "action" in h5_file and "gripper_position" in h5_file["action"]:
            data = h5_file["action"]["gripper_position"][:]
        elif "action/gripper_position" in h5_file:
            data = h5_file["action/gripper_position"][:]
        else:
            return None
        
        # Handle scalar or array formats
        data = np.asarray(data)
        if data.ndim == 0:
            data = np.array([data])
        elif data.ndim > 1:
            # Flatten if needed, but warn
            data = data.flatten()
        
        # Validate range
        if np.any(data < 0) or np.any(data > 1):
            print(f"Warning: gripper_position values outside [0, 1] range: min={data.min()}, max={data.max()}")
        
        return data
    except (KeyError, AttributeError) as e:
        print(f"Error reading gripper_position: {e}")
        return None


def median_filter(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply median filter to signal using numpy.
    
    Args:
        signal: Input signal
        kernel_size: Window size (must be odd)
    
    Returns:
        Filtered signal
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    half = kernel_size // 2
    filtered = np.zeros_like(signal)
    
    for i in range(len(signal)):
        start = max(0, i - half)
        end = min(len(signal), i + half + 1)
        filtered[i] = np.median(signal[start:end])
    
    return filtered


def filter_signal(
    signal: np.ndarray, 
    median_window: int = 5, 
    min_duration: int = 3,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter the gripper position signal to remove noise and spikes.
    
    Args:
        signal: Raw gripper position signal (0-1)
        median_window: Window size for median filter (must be odd)
        min_duration: Minimum duration in timesteps for a state to be valid
        threshold: Threshold for binary conversion (0.5)
    
    Returns:
        Tuple of (filtered_signal, binary_signal)
    """
    # Ensure median window is odd
    if median_window % 2 == 0:
        median_window += 1
    
    # Apply median filter to remove spikes
    if len(signal) > median_window:
        filtered = median_filter(signal, kernel_size=median_window)
    else:
        filtered = signal.copy()
    
    # Convert to binary using threshold
    binary = (filtered >= threshold).astype(int)
    
    # Apply minimum duration filter
    # Remove state changes that don't last at least min_duration timesteps
    if min_duration > 1 and len(binary) > min_duration:
        filtered_binary = binary.copy()
        
        # Find transitions
        diff = np.diff(filtered_binary)
        transitions = np.where(diff != 0)[0]
        
        # Check each transition
        for trans_idx in transitions:
            new_state = filtered_binary[trans_idx + 1]
            
            # Check forward: how long does this state last?
            forward_duration = 0
            for i in range(trans_idx + 1, len(filtered_binary)):
                if filtered_binary[i] == new_state:
                    forward_duration += 1
                else:
                    break
            
            # Check backward: how long was the previous state?
            prev_state = filtered_binary[trans_idx]
            backward_duration = 0
            for i in range(trans_idx, -1, -1):
                if filtered_binary[i] == prev_state:
                    backward_duration += 1
                else:
                    break
            
            # If transition doesn't last long enough, revert it
            if forward_duration < min_duration:
                # Revert the transition
                filtered_binary[trans_idx + 1:] = prev_state
            elif backward_duration < min_duration:
                # Previous state was too short, keep the transition
                pass
        
        binary = filtered_binary
    
    return filtered, binary


def label_stages(
    binary_signal: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Label motion stages based on gripper state transitions.
    
    Stages:
    - PREGRASPING (0): Initial period where gripper is open
    - TRANSPORT (1): Middle period where gripper is closed
    - PLACE (2): Final period where gripper is open again
    - UNKNOWN (3): Unclassified stage
    
    Args:
        binary_signal: Binary signal (0=open, 1=closed)
        threshold: Threshold used (for reference)
    
    Returns:
        Array of stage labels as integers (Stage enum values)
    """
    stages = np.full(len(binary_signal), Stage.UNKNOWN, dtype=np.int32)
    
    # Find transitions
    diff = np.diff(binary_signal)
    transitions = np.where(diff != 0)[0]
    
    # Find first 0->1 transition (start of transport)
    first_close_idx = None
    for trans_idx in transitions:
        if binary_signal[trans_idx] == 0 and binary_signal[trans_idx + 1] == 1:
            first_close_idx = trans_idx + 1
            break
    
    # Find last 1->0 transition (start of place)
    last_open_idx = None
    for trans_idx in reversed(transitions):
        if binary_signal[trans_idx] == 1 and binary_signal[trans_idx + 1] == 0:
            last_open_idx = trans_idx + 1
            break
    
    # Label stages
    if first_close_idx is None:
        # No closing detected - all pregrasping
        stages[:] = Stage.PREGRASPING
    elif last_open_idx is None:
        # No opening after closing - pregrasping then transport
        stages[:first_close_idx] = Stage.PREGRASPING
        stages[first_close_idx:] = Stage.TRANSPORT
    else:
        # Normal case: pregrasping -> transport -> place
        stages[:first_close_idx] = Stage.PREGRASPING
        stages[first_close_idx:last_open_idx] = Stage.TRANSPORT
        stages[last_open_idx:] = Stage.PLACE
    
    return stages


def write_stages_to_hdf5(
    h5_filepath: Path,
    stages: np.ndarray,
    skip_existing: bool = False
) -> bool:
    """
    Write stage labels back to HDF5 file.
    
    Args:
        h5_filepath: Path to HDF5 file
        stages: Array of stage labels as integers (Stage enum values)
        skip_existing: If True, skip if stage dataset already exists
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with h5py.File(h5_filepath, "r+") as f:
            # Check if already exists
            stage_exists = False
            if "action" in f and "stage" in f["action"]:
                stage_exists = True
            
            if stage_exists:
                if skip_existing:
                    return False
                # Delete existing dataset
                del f["action"]["stage"]
            
            # Create action group if it doesn't exist
            if "action" not in f:
                f.create_group("action")
            
            # Create stage dataset - now storing as integers (much simpler!)
            f["action"].create_dataset("stage", data=stages, dtype=np.int32)

        return True
    except Exception as e:
        # # #region agent log
        # with open('/mnt/data2/droid/droid/.cursor/debug.log', 'a') as log_file:
        #     log_file.write(json.dumps({"sessionId": "debug-session", "runId": "pre-fix", "hypothesisId": "A", "location": "label_gripper_stages.py:264", "message": "write_stages_to_hdf5 error", "data": {"error": str(e), "error_type": type(e).__name__}, "timestamp": __import__('time').time() * 1000}) + '\n')
        # # #endregion
        print(f"Error writing stages to {h5_filepath}: {e}")
        return False


def plot_gripper_stages(
    gripper_position: np.ndarray,
    stages: np.ndarray,
    filtered_signal: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Generate visualization plot of gripper position and stages.
    
    Args:
        gripper_position: Raw gripper position signal
        stages: Stage labels as integers (Stage enum values)
        filtered_signal: Optional filtered signal to overlay
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    time = np.arange(len(gripper_position))
    
    # Plot 1: Gripper position
    ax1.plot(time, gripper_position, 'b-', alpha=0.5, label='Raw signal', linewidth=1)
    if filtered_signal is not None:
        ax1.plot(time, filtered_signal, 'g-', alpha=0.7, label='Filtered signal', linewidth=1.5)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax1.set_ylabel('Gripper Position')
    ax1.set_title('Gripper Position vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Stages
    # Create color map for stages
    stage_colors = {
        Stage.PREGRASPING: 'blue',
        Stage.TRANSPORT: 'orange',
        Stage.PLACE: 'green',
        Stage.UNKNOWN: 'gray'
    }
    
    stage_names = {
        Stage.PREGRASPING: 'pregrasping',
        Stage.TRANSPORT: 'transport',
        Stage.PLACE: 'place',
        Stage.UNKNOWN: 'unknown'
    }
    
    # Plot as colored regions
    unique_stages = np.unique(stages)
    for stage_int in unique_stages:
        mask = stages == stage_int
        if np.any(mask):
            color = stage_colors.get(stage_int, 'gray')
            stage_name = stage_names.get(stage_int, 'unknown')
            ax2.fill_between(time, 0, 1, where=mask, alpha=0.3, color=color, label=stage_name)
    
    # Also plot as line for clarity
    ax2.plot(time, stages, 'k-', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Stage')
    ax2.set_xlabel('Time (timesteps)')
    ax2.set_title('Motion Stage vs Time')
    ax2.set_yticks([Stage.PREGRASPING, Stage.TRANSPORT, Stage.PLACE, Stage.UNKNOWN])
    ax2.set_yticklabels(['pregrasping', 'transport', 'place', 'unknown'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 3.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_file(
    h5_filepath: Path,
    threshold: float = 0.5,
    median_window: int = 5,
    min_duration: int = 3,
    skip_existing: bool = False
) -> str:
    """
    Process a single HDF5 file: extract, filter, label, save, and plot.
    
    Returns:
        "success" if successful, "skipped" if skipped, "failed" if failed
    """
    try:
        # Read gripper position
        with h5py.File(h5_filepath, "r") as f:
            gripper_position = read_gripper_position(f)
        
        if gripper_position is None:
            print(f"Skipping {h5_filepath}: no action/gripper_position found")
            return "skipped"
        
        if len(gripper_position) < min_duration * 2:
            print(f"Skipping {h5_filepath}: trajectory too short ({len(gripper_position)} timesteps)")
            return "skipped"
        
        # Filter signal
        filtered_signal, binary_signal = filter_signal(
            gripper_position, 
            median_window=median_window,
            min_duration=min_duration,
            threshold=threshold
        )
        
        # Label stages
        stages = label_stages(binary_signal, threshold=threshold)
        
        # Write back to HDF5
        success = write_stages_to_hdf5(h5_filepath, stages, skip_existing=skip_existing)
        if not success:
            if skip_existing:
                print(f"Skipping {h5_filepath}: stage dataset already exists")
                return "skipped"
            return "failed"
        
        # Generate plot
        plot_filename = h5_filepath.stem + "_gripper_stages.png"
        plot_path = h5_filepath.parent / plot_filename
        plot_gripper_stages(
            gripper_position,
            stages,
            filtered_signal=filtered_signal,
            output_path=plot_path
        )
        
        return "success"
        
    except Exception as e:
        print(f"Error processing {h5_filepath}: {e}")
        import traceback
        traceback.print_exc()
        return "failed"


def main(
    data_dir: str,
    threshold: float = 0.5,
    median_window: int = 5,
    min_duration: int = 3,
    skip_existing: bool = False,
):
    """
    Main function to process all HDF5 files in the given directory.
    
    Args:
        data_dir: Root directory containing HDF5 files (searched recursively)
        threshold: Threshold for binary conversion (0.5)
        median_window: Window size for median filter (must be odd, default 5)
        min_duration: Minimum duration in timesteps for a state to be valid (default 3)
        skip_existing: If True, skip files that already have stage dataset
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Find all HDF5 files
    h5_files = find_hdf5_files(data_path)
    print(f"Found {len(h5_files)} HDF5 files")
    
    if len(h5_files) == 0:
        print("No HDF5 files found. Exiting.")
        return
    
    # Process each file
    successful = 0
    failed = 0
    skipped = 0
    
    for h5_file in tqdm(h5_files, desc="Processing files"):
        result = process_file(
            h5_file,
            threshold=threshold,
            median_window=median_window,
            min_duration=min_duration,
            skip_existing=skip_existing
        )
        
        if result == "success":
            successful += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    tyro.cli(main)

